from typing import Dict, List
import torch
from torch_scatter import scatter_mean
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn
from torch.distributions import Exponential
from pathlib import Path
from rdkit import Chem
from src.data_processing.geom import MoleculeFeaturizer

from src.models.lr_scheduler import LRScheduler
from src.models.interpolant_scheduler import InterpolantScheduler
from src.models.vector_field import CTMCVectorField, ContextualCTMCVectorField
from src.models.utils import *
from src.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from src.data_processing.priors import inference_prior_register, edge_prior
from src.analysis.molecule_builder import SampledMolecule
# from src.analysis.metrics import SampleAnalyzer
from einops import rearrange

class FlowMol(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']
    node_feats = ['x', 'a', 'c']
    edge_feats = ['e']

    def __init__(self,
                 atom_type_map: List[str],               
                 n_atoms_hist_file: str,
                 marginal_dists_file: str,                 
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5,
                 sample_interval: float = 1.0, # how often to sample molecules from the model, measured in epochs
                 n_mols_to_sample: int = 64, # how many molecules to sample from the model during each sample/eval step during training
                 time_scaled_loss: bool = True,
                 exclude_charges: bool = False,
                 weight_ae: bool = False, # whether or not to apply weights to the atom and edge losses (infrequent categories given more weight)
                 target_blur: float = 0.0, # how much to blur the target distribution for categorical features
                 total_loss_weights: Dict[str, float] = {}, 
                 lr_scheduler_config: dict = {},
                 interpolant_scheduler_config: dict = {},
                 vector_field_config: dict = {},
                 prior_config: dict = {},
                 default_n_timesteps: int = 250,
                 explicit_aromaticity: bool = False
                 ):
        super().__init__()

        self.lr_scheduler_config = lr_scheduler_config
        self.atom_type_map = atom_type_map
        self.n_atom_types = len(atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types if explicit_aromaticity else n_bond_types - 1
        self.total_loss_weights = total_loss_weights
        self.time_scaled_loss = time_scaled_loss
        self.prior_config = prior_config
        self.exclude_charges = exclude_charges
        self.marginal_dists_file = marginal_dists_file
        self.weight_ae = weight_ae
        self.target_blur = target_blur
        self.n_atoms_hist_file = n_atoms_hist_file
        self.default_n_timesteps = default_n_timesteps
        self.explicit_aromaticity = explicit_aromaticity

        if self.prior_config.get('x', {}).get('scaling_factor'):
            self.total_loss_weights['x'] = self.prior_config['x']['scaling_factor']**2
            vector_field_config['rbf_dmax'] = vector_field_config['rbf_dmax'] / self.prior_config['x']['scaling_factor']
        
        if self.target_blur < 0.0:
            raise ValueError('target_blur must be non-negative')
        
        # if provided filepath to data dir does not exist, assume it is relative to the repo root
        processed_data_dir = Path(self.marginal_dists_file).parent
        if not processed_data_dir.exists():
            repo_root = Path(__file__).parent.parent.parent
            self.marginal_dists_file = repo_root / self.marginal_dists_file
            self.n_atoms_hist_file = repo_root / self.n_atoms_hist_file

        if self.exclude_charges:
            self.node_feats.remove('c')
            self.canonical_feat_order.remove('c')
            self.total_loss_weights.pop('c', None)

        # create a dictionary mapping feature -> number of categories
        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': self.n_atom_charges,
            'e': self.n_bond_types,
        }

        for feat in self.canonical_feat_order:
            if feat not in total_loss_weights:
                self.total_loss_weights[feat] = 1.0

                # print warning if the user has not specified a loss weight for a feature
                print(f'WARNING: no loss weight specified for feature {feat}, using default of 1.0')

        self.exp_dist = Exponential(1.0)
        
        # construct histogram of number of atoms in each ligand
        self.build_n_atoms_dist(n_atoms_hist_file=self.n_atoms_hist_file)

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(canonical_feat_order=self.canonical_feat_order, 
                                                          **interpolant_scheduler_config)
        
        self.vector_field = CTMCVectorField(n_atom_types=self.n_atom_types,
                                        canonical_feat_order=self.canonical_feat_order,
                                        interpolant_scheduler=self.interpolant_scheduler, 
                                        n_charges=n_atom_charges, 
                                        n_bond_types=self.n_bond_types,
                                        exclude_charges=self.exclude_charges,
                                        **vector_field_config)

        self.sample_interval = sample_interval # how often to sample molecules from the model, measured in epochs
        self.n_mols_to_sample = n_mols_to_sample # how many molecules to sample from the model during each sample/eval step during training
        self.last_sample_marker = 0 # this is the epoch_exact value of the last time we sampled molecules from the model
        # self.sample_analyzer = SampleAnalyzer()

        # record the last epoch value for training steps -  this is really hacky but it lets me
        # align the validation losses with the correspoding training epoch value on W&B
        self.last_epoch_exact = 0

        self.save_hyperparameters()

        try:
            p_a, p_c, p_e, p_c_given_a = torch.load(self.marginal_dists_file)
        except:
            p_a, p_c, p_e, p_c_given_a, p_r, r_bin_edges = torch.load(self.marginal_dists_file)
        self.p_a = p_a
        self.p_e = p_e


                
    def configure_loss_fns(self, device):    
        # instantiate loss functions
        reduction = 'none'

        categorical_loss_fn = nn.CrossEntropyLoss


        if self.weight_ae:
            p_a_max = torch.max(self.p_a)
            p_e_max = torch.max(self.p_e)
            p_a_safe = self.p_a.clamp(min=0.001)
            p_e_safe = self.p_e.clamp(min=0.001)
            a_kwargs = {'weight': (p_a_max/p_a_safe**0.3).to(device)}
            e_kwargs = {'weight': (p_e_max/p_e_safe**0.3).to(device)}
        else:
            a_kwargs = {}
            e_kwargs = {}


        cat_kwargs = {'ignore_index': -100}


        self.loss_fn_dict = {
            'x': nn.MSELoss(reduction=reduction),
            'a': categorical_loss_fn(reduction=reduction, **a_kwargs, **cat_kwargs),
            'c': categorical_loss_fn(reduction=reduction, **cat_kwargs),
            'e': categorical_loss_fn(reduction=reduction, **e_kwargs, **cat_kwargs),
        }

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        # check if self has the attribute batches_per_epoch
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        # compute epoch as a float
        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.last_epoch_exact = epoch_exact

        # update the learning rate
        self.lr_scheduler.step_lr(epoch_exact)

        # sample and evaluate molecules if necessary
        if epoch_exact - self.last_sample_marker >= self.sample_interval:
            self.last_sample_marker = epoch_exact
            self.eval()
            with torch.no_grad():
                sampled_molecules = self.sample_random_sizes(n_molecules=self.n_mols_to_sample, device=g.device)
            self.train()
            # sampled_mols_metrics = self.sample_analyzer.analyze(sampled_molecules, energy_div=False, functional_validity=True)
            # self.log_dict(sampled_mols_metrics)

        # compute losses
        losses = self(g)

        # create a dictionary of values to log
        train_log_dict = {}
        train_log_dict['epoch_exact'] = epoch_exact

        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]
            self.log(f'{feat}_loss', losses[feat], prog_bar=True, on_step=True, sync_dist=True)

        self.log_dict(train_log_dict, sync_dist=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, sync_dist=True)

        return total_loss
    
    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        # compute losses
        losses = self(g)

        # create dictionary of values to log
        val_log_dict = {
            'epoch_exact': self.last_epoch_exact
        }

        for key in losses:
            val_log_dict[f'{key}_val_loss'] = losses[key]

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        # combine individual losses into a total loss
        total_loss = torch.zeros(1, device=g.device, requires_grad=False)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log('val_total_loss', total_loss, prog_bar=True, batch_size=g.batch_size, on_step=True, sync_dist=True)

        return total_loss
    
    def forward(self, g: dgl.DGLGraph):
        
        batch_size = g.batch_size
        device = g.device

        # check if the attribute loss_fn_dict exists
        # it is necessary to do this here (as opposed to in __init__) beacause
        # to instantiate the loss function class-conditioned weights, the weight
        # tensors need to be on the same device as the graph...seems pretty dumb but that's how it is
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)
        
        # remove charge loss function if necessary
        if self.exclude_charges:
            self.loss_fn_dict.pop('c', None)

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # sample timepoints for each molecule in the batch
        t = torch.rand(batch_size, device=device).float()
        


        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)
        if hasattr(self.vector_field, 'enable_dynamic_graph') and self.vector_field.enable_dynamic_graph:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.vector_field.knn_connectivity)
            

        vf_output = self.vector_field(g, t, node_batch_idx=node_batch_idx, edge_batch_idx=edge_batch_idx, upper_edge_mask=upper_edge_mask)

        # forward pass for the vector field

        # get the target (label) for each feature
        targets = {}
        
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # sigma_t_i = self.interpolant_scheduler.sigma_t_prime(t, self.vector_field.eta)[:, feat_idx][node_batch_idx].unsqueeze(-1)
            # sigma_t_prime_i = self.interpolant_scheduler.sigma_t_prime(t, self.vector_field.eta)[:, feat_idx][node_batch_idx].unsqueeze(-1)
            # alpha_t_i = self.interpolant_scheduler.alpha_t(t)[:, feat_idx][node_batch_idx].unsqueeze(-1)
            # alpha_t_prime_i = self.interpolant_scheduler.alpha_t_prime(t)[:, feat_idx][node_batch_idx].unsqueeze(-1)

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata


            # else:
            target = data_src[f'{feat}_1_true']
            
            if feat == "e":
                target = target[upper_edge_mask]
            if feat in ['a', 'c', 'e']:
                if self.target_blur == 0.0:
                    target = target.argmax(dim=-1)
                else:
                    target = target + torch.randn_like(target)*self.target_blur
                    target = fn.softmax(target, dim=-1)
                    target = target.argmax(dim=-1)

                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0
            
            # sample weight
            # if 'sample_weights' in g.ndata:
            #     sample_weights = g.ndata['sample_weights']
            #     if feat == 'e':
            #         sample_weights = scatter_mean(sample_weights, node_batch_idx, dim=0) 
            #         sample_weights = sample_weights[edge_batch_idx][upper_edge_mask]
            # else:
            #     sample_weights = 1.0

            # compute the losses
            target = targets[feat]
            # if feat == 'x':
            #     vector_pred = vf_output[feat] - g.ndata['x_0']
            #     losses[feat] = self.loss_fn_dict[feat](vector_pred, target)*weight
            # else:
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)*weight

            # when time_scaled_loss is True, we set the reduction to 'none' so that each training example can be scaled by the time-dependent weight.
            # however, this means that we also need to do the reduction ourselves here.
            if feat == 'x':
                losses[feat] = losses[feat].mean()
            else:
                losses[feat] = losses[feat][target != -100].mean()

        return losses
    
    def sample_prior(self, g, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        """Sample from the prior distribution of the ligand."""
        # sample atom positions from prior
        # TODO: we should set the standard deviation of atom position prior to be like the average distance to the COM in the training set
        # or perhaps the average distance to COM for molecules with the same number of atoms
        num_nodes = g.num_nodes()
        device = g.device

        
        # sample the prior for node features
        for feat in self.node_feats:
            if feat == 'x':
                prior_fn = inference_prior_register['centered-normal']
            else:
                prior_fn = inference_prior_register['ctmc']
            # I tried to design consistent interface for prior functions, but it's not perfect
            # hence the need for the following two if statements
            if feat == 'x':
                args = [g, node_batch_idx,]
            else:
                args = [num_nodes, self.n_cat_dict[feat],]

            kwargs = self.prior_config[feat]['kwargs']
            g.ndata[f'{feat}_0'] = prior_fn(*args, **kwargs).to(device)

        # sample the prior for edge features
        g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e'], explicit_aromaticity=self.explicit_aromaticity).to(device)
            
        return g
    

    def configure_optimizers(self):
        try:
            weight_decay = self.lr_scheduler_config['weight_decay']
        except KeyError:
            weight_decay = 0

        optimizer = optim.Adam(self.parameters(), lr=self.lr_scheduler_config['base_lr'], weight_decay=weight_decay)
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int, **kwargs):
        """Draw samples from the distribution of the number of atoms in a ligand."""
        n_atoms = self.n_atoms_dist.sample((n_molecules,), **kwargs)
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
                                stochasticity=None, high_confidence_threshold=None, 
                                xt_traj=False, ep_traj=False, **kwargs):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        return self.sample(atoms_per_molecule, 
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj, **kwargs)
    

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False, **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)
        

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'stochasticity': stochasticity,
            'high_confidence_threshold': high_confidence_threshold
        }

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames, upper_edge_mask = itg_result

        elif isinstance(itg_result, tuple):
            g, upper_edge_mask = itg_result
        else:
            g = itg_result

        if self.prior_config.get('x', {}).get('scaling_factor'):
            g.ndata['x_1'] = g.ndata['x_1'] * self.prior_config['x']['scaling_factor']


        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        ctmc_mol = True

        molecules = []

        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges,
                explicit_aromaticity=self.explicit_aromaticity))

        return molecules

    @torch.no_grad()
    def sample_with_frag(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False, frag_file = None, **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        mol_frag = Chem.SDMolSupplier(str(frag_file), removeHs=True, sanitize=True)
        n_bond_orders = 5 if self.explicit_aromaticity else 4
        mol_featurizer = MoleculeFeaturizer(self.atom_type_map, explicit_aromaticity=self.explicit_aromaticity)
        positions, atom_types, atom_charges, bond_types, bond_idxs, failed_idx, bond_order_counts = mol_featurizer.featurize_molecules(mol_frag)
        frag_features = {
            'x': positions,
            'a': atom_types,
            'c': atom_charges,
            'e': bond_types,
            'e_pair': bond_idxs
        }

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'stochasticity': stochasticity,
            'high_confidence_threshold': high_confidence_threshold,
            'frag_features': frag_features
        }

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames, upper_edge_mask = itg_result

        elif isinstance(itg_result, tuple):
            g, upper_edge_mask = itg_result
        else:
            g = itg_result

        if self.prior_config.get('x', {}).get('scaling_factor'):
            g.ndata['x_1'] = g.ndata['x_1'] * self.prior_config['x']['scaling_factor']


        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        ctmc_mol = True

        molecules = []

        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges,
                explicit_aromaticity=self.explicit_aromaticity))

        return molecules

class PocketFlowMol(FlowMol): # 继承自FlowMol以复用一些方法
    
    def __init__(self,
                 # --- 新增的参数 ---
                 pretrain_config: dict,
                 **kwargs # 捕获其他FlowMol参数
                 ):
        
        # 调用父类的构造函数，但先不创建vector_field
        # 我们用一个假的vector_field_config来绕过父类的初始化
        super().__init__(**kwargs)
        
        # 创建一个动态的 ContextualVectorField 类，它继承自正确的基类
        vector_field_class = ContextualCTMCVectorField
        
        self.vector_field = vector_field_class(
            # --- 传递给基类VectorField的参数 ---
            n_atom_types=self.n_atom_types,
            canonical_feat_order=self.canonical_feat_order,
            interpolant_scheduler=self.interpolant_scheduler, 
            n_charges=self.n_atom_charges, 
            n_bond_types=self.n_bond_types,
            exclude_charges=self.exclude_charges,
            **kwargs.get('vector_field_config', {})
        )
        
        self.build_n_atoms_dist(n_atoms_hist_file=self.n_atoms_hist_file)

        if pretrain_config['pretrain_path']:
            pretrain_model = torch.load(pretrain_config['pretrain_path'])['state_dict']
            is_freeze = pretrain_config['is_freeze']
            self.load_pretrain_model(pretrain_model, is_freeze = is_freeze)


    def forward(self, g: dgl.DGLGraph, pocket_g: dgl.DGLGraph):
        # 这个方法现在接收两个图
        
        batch_size = g.batch_size
        device = g.device

        # check if the attribute loss_fn_dict exists
        # it is necessary to do this here (as opposed to in __init__) beacause
        # to instantiate the loss function class-conditioned weights, the weight
        # tensors need to be on the same device as the graph...seems pretty dumb but that's how it is
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        upper_edge_mask = get_upper_edge_mask(g)

        t = torch.rand(batch_size, device=device).float()

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)
        if hasattr(self.vector_field, 'enable_dynamic_graph') and self.vector_field.enable_dynamic_graph:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.vector_field.knn_connectivity)
            

        vf_output = self.vector_field(g, pocket_g, t, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # compute the target for endpoint parameterization
            target = data_src[f'{feat}_1_true']
            if feat == "e":
                target = target[upper_edge_mask]
            if feat in ['a', 'c', 'e']:
                if self.target_blur == 0.0:
                    target = target.argmax(dim=-1)
                else:
                    target = target + torch.randn_like(target)*self.target_blur
                    target = fn.softmax(target, dim=-1)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
            
        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)*weight

            # when time_scaled_loss is True, we set the reduction to 'none' so that each training example can be scaled by the time-dependent weight.
            # however, this means that we also need to do the reduction ourselves here.
            if feat == 'x':
                losses[feat] = losses[feat].mean()
            else:
                losses[feat] = losses[feat][target != -100].mean()

        return losses

    def training_step(self, batch, batch_idx: int):
        # batch 现在是一个元组 (ligand_g, pocket_g)
        g, pocket_g = batch
        
        # check if self has the attribute batches_per_epoch
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        # compute epoch as a float
        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.last_epoch_exact = epoch_exact

        # update the learning rate
        self.lr_scheduler.step_lr(epoch_exact)

        # sample and evaluate molecules if necessary
        if epoch_exact - self.last_sample_marker >= self.sample_interval:
            self.last_sample_marker = epoch_exact
            self.eval()
            with torch.no_grad():
                n_molecules = pocket_g.batch_size
                sampled_molecules = self.sample_random_sizes(n_molecules=n_molecules, pocket_g = pocket_g, device=g.device)
            self.train()
            # sampled_mols_metrics = self.sample_analyzer.analyze(sampled_molecules, energy_div=False, functional_validity=True)
            # self.log_dict(sampled_mols_metrics)

        # compute losses
        losses = self(g, pocket_g)

        # create a dictionary of values to log
        train_log_dict = {}
        train_log_dict['epoch_exact'] = epoch_exact

        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]
            self.log(f'{feat}_loss', losses[feat], prog_bar=True, on_step=True, sync_dist=True)

        self.log_dict(train_log_dict, sync_dist=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx: int):
        g, pocket_g = batch
        # compute losses
        losses = self(g, pocket_g)

        # create dictionary of values to log
        val_log_dict = {
            'epoch_exact': self.last_epoch_exact
        }

        for key in losses:
            val_log_dict[f'{key}_val_loss'] = losses[key]

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        # combine individual losses into a total loss
        total_loss = torch.zeros(1, device=g.device, requires_grad=False)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log('val_total_loss', total_loss, prog_bar=True, batch_size=g.batch_size, on_step=True, sync_dist=True)

        return total_loss
    
    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the joint distribution of ligand and pocket atoms."""
        # Load joint distribution data
        joint_dist_data = torch.load(n_atoms_hist_file)
        unique_pairs = joint_dist_data['unique_pairs']
        pair_counts = joint_dist_data['counts']
        
        # Create joint probability distribution
        joint_probs = pair_counts.float() / pair_counts.sum()
        self.joint_dist = torch.distributions.Categorical(probs=joint_probs)
        self.unique_pairs = unique_pairs
        
        # Precompute conditional distributions
        self._build_conditional_dists()

    def _build_conditional_dists(self):
        """Precompute conditional distributions for each pocket size."""
        self.conditional_dists = {}
        unique_pocket_sizes = torch.unique(self.unique_pairs[:, 1])
        
        for p_size in unique_pocket_sizes:
            mask = (self.unique_pairs[:, 1] == p_size)
            indices = mask.nonzero().squeeze(-1)
            cond_probs = self.joint_dist.probs[indices]
            
            if cond_probs.sum() > 0:
                cond_probs = cond_probs / cond_probs.sum()  # Normalize
                self.conditional_dists[p_size.item()] = {
                    'indices': indices,
                    'dist': torch.distributions.Categorical(probs=cond_probs)
                }

    def sample_n_atoms(self, n_molecules: int, pocket_g: dgl.DGLGraph, **kwargs):
        """Draw conditional samples based on pocket sizes."""
        # Get pocket sizes (number of atoms per pocket)
        if pocket_g.batch_size > 1:
            pocket_sizes = pocket_g.batch_num_nodes()  # For batched graphs
        else:
            pocket_sizes = torch.tensor(
                [pocket_g.num_nodes()], 
                device=pocket_g.device
            )
        
        samples = []
        for p_size in pocket_sizes:
            p_size_val = p_size.item()
            
            # Try conditional sampling first
            if p_size_val in self.conditional_dists:
                cond_dist = self.conditional_dists[p_size_val]
                cond_samples = cond_dist['dist'].sample((n_molecules,))
                global_indices = cond_dist['indices'][cond_samples]
            else:
                # Fallback to global sampling with pocket size replacement
                global_indices = self.joint_dist.sample((n_molecules,))
                ligand_sizes = self.unique_pairs[global_indices, 0]
                global_indices = torch.stack([
                    ligand_sizes,
                    torch.full((n_molecules,), p_size_val)
                ], dim=1)
                # Need to find closest matching indices in unique_pairs
                # This is simplified - may need more sophisticated matching
                distances = torch.norm(
                    self.unique_pairs.float() - global_indices.float().unsqueeze(1),
                    dim=2
                )
                global_indices = distances.argmin(dim=1)
            
            samples.append(self.unique_pairs[global_indices])
        
        return torch.cat(samples, dim=0)

    def sample_random_sizes(self, n_molecules: int, pocket_g: dgl.DGLGraph, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, 
        xt_traj=False, ep_traj=False, **kwargs):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(1, pocket_g).to(device) # sample 1 molecule for each pocket

        return self.sample(atoms_per_molecule[:,0], pocket_g,
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj, **kwargs)

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, pocket_g: dgl.DGLGraph, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False, **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)
        

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'stochasticity': stochasticity,
            'high_confidence_threshold': high_confidence_threshold
        }

        itg_result = self.vector_field.integrate(g, pocket_g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames, upper_edge_mask = itg_result

        elif isinstance(itg_result, tuple):
            g, upper_edge_mask = itg_result
        else:
            g = itg_result

        if self.prior_config.get('x', {}).get('scaling_factor'):
            g.ndata['x_1'] = g.ndata['x_1'] * self.prior_config['x']['scaling_factor']


        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        ctmc_mol = True



        molecules = []

        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges,
                explicit_aromaticity=self.explicit_aromaticity))

        return molecules
    
    @torch.no_grad()
    def sample_with_frag(self, n_atoms: torch.Tensor, pocket_g: dgl.DGLGraph, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False, frag_file = None, **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)
        mol_frag = Chem.SDMolSupplier(str(frag_file), removeHs=True, sanitize=True)
        n_bond_orders = 5 if self.explicit_aromaticity else 4
        mol_featurizer = MoleculeFeaturizer(self.atom_type_map, explicit_aromaticity=self.explicit_aromaticity)
        positions, atom_types, atom_charges, bond_types, bond_idxs, failed_idx, bond_order_counts = mol_featurizer.featurize_molecules(mol_frag)
        frag_features = {
            'x': positions,
            'a': atom_types,
            'c': atom_charges,
            'e': bond_types,
            'e_pair': bond_idxs
        }

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'stochasticity': stochasticity,
            'high_confidence_threshold': high_confidence_threshold,
            'frag_features': frag_features
        }

        itg_result = self.vector_field.integrate(g, pocket_g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames, upper_edge_mask = itg_result

        elif isinstance(itg_result, tuple):
            g, upper_edge_mask = itg_result
        else:
            g = itg_result

        if self.prior_config.get('x', {}).get('scaling_factor'):
            g.ndata['x_1'] = g.ndata['x_1'] * self.prior_config['x']['scaling_factor']


        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        ctmc_mol = True



        molecules = []

        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges,
                explicit_aromaticity=self.explicit_aromaticity))

        return molecules

    def load_pretrain_model(self, state_dict: dict, is_freeze: bool = False, strict: bool = True):
        """Load state dict with exact name and shape matching.
        
        Args:
            state_dict: Dictionary containing parameters and persistent buffers.
            strict: If True, requires exact matching of keys and shapes.
                If False, ignores non-matching keys.
        """
        model_dict = self.state_dict()
        matched, unmatched, shape_mismatch = [], [], []

        # Create mapping from conv_layers to hetero_conv_layers
        def map_key(key):
            if not key.startswith('vector_field.conv_layers.'):
                return key
                
            parts = key.split('.')
            layer_num = int(parts[2].split('_')[-1])
            remaining_path = '.'.join(parts[3:])
            remaining_path_speacial = '.'.join(parts[4:])
        
            # Handle edge message components
            if 'edge_message' in remaining_path:
                return f'vector_field.hetero_conv_layers.{layer_num}.edge_message.ligand_ll_ligand.{remaining_path_speacial}'
            
            # Handle node update components
            elif 'node_update' in remaining_path:
                return f'vector_field.hetero_conv_layers.{layer_num}.node_update.ligand.{remaining_path_speacial}'
            
            else:
                return f'vector_field.hetero_conv_layers.{layer_num}.{remaining_path}'
            
            return key
        
        total_pretrain_params = len(state_dict)
        total_params = len(model_dict)
        used_target_keys = set()
            
        for key, val in state_dict.items():
            target = map_key(key)
            
            if target in model_dict.keys() and val.shape == model_dict[target].shape:
                model_dict[target] = val
                matched.append(target)
                used_target_keys.add(target)
                if is_freeze:
                    parts = target.split('.')
                    obj = self
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    getattr(obj, parts[-1]).requires_grad_(False)

            elif target in model_dict.keys():
                shape_mismatch.append(target)

            else:
                unmatched.append(target)
                
        self.load_state_dict(model_dict, strict=False)

        all_model_keys = set(model_dict.keys())
        missing_in_pretrain = all_model_keys - used_target_keys
        print(f"Missing in pretrain (need initialization): {len(missing_in_pretrain)} ({len(missing_in_pretrain)/total_params*100:.1f}%)")

        # 打印详细的迁移统计信息
        print("=" * 60)
        print("pretrain model parameter transfer statistics:")
        print("=" * 60)
        print(f"pretrain parameters total: {total_pretrain_params}")
        print(f"matched parameters: {len(matched)} ({len(matched)/total_pretrain_params*100:.1f}%)")
        print(f"unmatched parameters: {len(unmatched)} ({len(unmatched)/total_pretrain_params*100:.1f}%)")
        print(f"shape mismatch parameters: {len(shape_mismatch)} ({len(shape_mismatch)/total_pretrain_params*100:.1f}%)")
        print("-" * 60)
        
        if matched:
            # 按模块分类统计成功匹配的参数
            module_stats = {}
            for key in matched:
                module_name = key.split('.')[2] if len(key.split('.')) > 2 else 'other'
                module_stats[module_name] = module_stats.get(module_name, 0) + 1
            
            print("matched parameters distribution:")
            for module, count in sorted(module_stats.items()):
                print(f"  {module}: {count} parameters")
        
        if missing_in_pretrain:
            # 按模块分类统计缺失的参数
            missing_stats = {}
            for key in missing_in_pretrain:
                module_name = key.split('.')[2] if len(key.split('.')) > 2 else 'other'
                missing_stats[module_name] = missing_stats.get(module_name, 0) + 1
            
            print("MISSING PARAMETERS IN PRETRAIN (need initialization):")
            for module, count in sorted(missing_stats.items()):
                print(f"  {module}: {count} parameters")
            
            # 显示每个模块的具体参数示例
            print("\nDETAILED MISSING PARAMETERS BY MODULE:")
            for module in sorted(missing_stats.keys()):
                module_params = [k for k in missing_in_pretrain if k.split('.')[2] == module]
                print(f"\n{module.upper()} (total: {len(module_params)}):")
                for param in sorted(module_params)[:5]:  # 每个模块显示前5个
                    print(f"    {param}")
                if len(module_params) > 5:
                    print(f"    ... and {len(module_params) - 5} more")
            print("-" * 80)
        
        if unmatched:
            print(f"Warning: Unmatched keys: {unmatched}")
        if shape_mismatch:
            print(f"Warning: Shape mismatches: {shape_mismatch}")
        
        if strict and (unmatched or shape_mismatch):
            raise RuntimeError(
                f"Error loading state_dict:\n"
                f"Unmatched: {unmatched}\n"
                f"Shape mismatches: {shape_mismatch}"
            )
        
        return matched, unmatched, shape_mismatch