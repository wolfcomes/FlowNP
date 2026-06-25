from src.models.flowmol import FlowMol, PocketFlowMol
from pathlib import Path
import yaml
from src.data_processing.data_module import MoleculeDataModule, PocketLigandDataModule

def read_config_file(config_file: Path) -> dict:
    # process config file into dictionary
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def model_from_config(config: dict, seed_ckpt: Path = None) -> FlowMol:

    atom_type_map = config['dataset']['atom_map']

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']
    explicit_aromaticity = config['dataset']['explicit_aromaticity']


    # get the filepath of the n_atoms histogram
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_n_atoms_histogram.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    if seed_ckpt is not None:
        model = FlowMol.load_from_checkpoint(seed_ckpt, 
                                            atom_type_map=atom_type_map,
                                            n_atoms_hist_file=n_atoms_hist_filepath,
                                            marginal_dists_file=marginal_dists_file,
                                            sample_interval=sample_interval,
                                            n_mols_to_sample=mols_to_sample,
                                            explicit_aromaticity=explicit_aromaticity,
                                            vector_field_config=config['vector_field'],
                                            interpolant_scheduler_config=config['interpolant_scheduler'], 
                                            lr_scheduler_config=config['lr_scheduler'],
                                            **config['mol_fm'])
    else:
        model = FlowMol(atom_type_map=atom_type_map, 
                        n_atoms_hist_file=n_atoms_hist_filepath,
                        marginal_dists_file=marginal_dists_file,
                        sample_interval=sample_interval,
                        n_mols_to_sample=mols_to_sample,
                        explicit_aromaticity=explicit_aromaticity,
                        vector_field_config=config['vector_field'],
                        interpolant_scheduler_config=config['interpolant_scheduler'], 
                        lr_scheduler_config=config['lr_scheduler'],
                        **config['mol_fm'])
    
    return model

def model_from_config(config: dict, seed_ckpt: Path = None) -> PocketFlowMol:

    atom_type_map = config['dataset']['atom_map']

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']
    pretrain_config = config['pretrain_config']
    explicit_aromaticity = config['dataset']['explicit_aromaticity']



    # get the filepath of the n_atoms histogram
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_joint_atoms_distribution.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    if seed_ckpt is not None:
        model = PocketFlowMol.load_from_checkpoint(seed_ckpt, 
                                            pretrain_config,
                                            atom_type_map=atom_type_map,
                                            n_atoms_hist_file=n_atoms_hist_filepath,
                                            marginal_dists_file=marginal_dists_file,
                                            sample_interval=sample_interval,
                                            n_mols_to_sample=mols_to_sample,
                                            explicit_aromaticity=explicit_aromaticity,
                                            vector_field_config=config['vector_field'],
                                            interpolant_scheduler_config=config['interpolant_scheduler'], 
                                            lr_scheduler_config=config['lr_scheduler'],
                                            **config['mol_fm'])
    else:
        model = PocketFlowMol(pretrain_config = pretrain_config,
                        atom_type_map=atom_type_map, 
                        n_atoms_hist_file=n_atoms_hist_filepath,
                        marginal_dists_file=marginal_dists_file,
                        sample_interval=sample_interval,
                        n_mols_to_sample=mols_to_sample,
                        explicit_aromaticity=explicit_aromaticity,
                        vector_field_config=config['vector_field'],
                        interpolant_scheduler_config=config['interpolant_scheduler'], 
                        lr_scheduler_config=config['lr_scheduler'],
                        **config['mol_fm'])
    
    return model

def data_module_from_config(config: dict) -> MoleculeDataModule:
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # determine if we are doing distributed training
    if config['training']['trainer_args']['devices'] > 1:
        distributed = True
    else:
        distributed = False

    data_module = MoleculeDataModule(dataset_config=config['dataset'],
                                     dm_prior_config=config['mol_fm']['prior_config'],
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     distributed=distributed)
    
    return data_module

def data_module_from_config(config: dict) -> PocketLigandDataModule:
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # determine if we are doing distributed training
    if config['training']['trainer_args']['devices'] > 1:
        distributed = True
    else:
        distributed = False

    data_module = PocketLigandDataModule(dataset_config=config['dataset'],
                                     dm_prior_config=config['mol_fm']['prior_config'],
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     distributed=distributed)
    
    return data_module