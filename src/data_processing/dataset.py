import torch
from pathlib import Path
import dgl
from torch.nn.functional import one_hot
from src.data_processing.priors import coupled_node_prior, edge_prior

# create a function named collate that takes a list of samples from the dataset and combines them into a batch
# this might not be necessary. I think we can pass the argument collate_fn=dgl.batch to the DataLoader
def collate(graphs):
    return dgl.batch(graphs)

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(MoleculeDataset, self).__init__()

        # unpack some configs regarding the prior
        self.prior_config = prior_config
        self.dataset_config = dataset_config
        self.explicit_aromaticity = dataset_config['explicit_aromaticity']
        self.n_bond_types = 5 if self.explicit_aromaticity else 4

        # get the processed data directory
        processed_data_dir: Path = Path(dataset_config['processed_data_dir'])

        # if split == 'train':
        #     weights_file = processed_data_dir / 'train_molecular_weights.pt'  # 或者你的权重文件路径
        #     if weights_file.exists():
        #         self.sample_weights = torch.load(weights_file)['weights']
        # else:
        #     print(f"{split}Using uniform weights.")
        #     self.sample_weights = None

        # if the processed data directory does not exist, check it relative to the root of flowmol repository
        if not processed_data_dir.exists():
            processed_data_dir = Path(__file__).parent.parent.parent / processed_data_dir
            if processed_data_dir.exists():
                dataset_config['processed_data_dir'] = str(processed_data_dir)
            else:
                raise FileNotFoundError(f"processed data directory {dataset_config['processed_data_dir']} not found.")
            
        self.processed_data_dir = processed_data_dir

        

        if dataset_config['dataset_name'] in ['geom', 'qm9', 'geom_5conf','coconut','crossdock']:
            data_file = processed_data_dir / f'{split}_data_processed.pt'
        else:
            raise NotImplementedError('unsupported dataset_name')

        # load data from processed data directory
        data_dict = torch.load(data_file)

        self.positions = data_dict['positions']
        self.atom_types = data_dict['atom_types']
        self.atom_charges = data_dict['atom_charges']
        self.bond_types = data_dict['bond_types']
        self.bond_idxs = data_dict['bond_idxs']
        self.node_idx_array = data_dict['node_idx_array']
        self.edge_idx_array = data_dict['edge_idx_array']

    def __len__(self):
        return self.node_idx_array.shape[0]
    
    def __getitem__(self, idx):
        node_start_idx = self.node_idx_array[idx, 0]
        node_end_idx = self.node_idx_array[idx, 1]
        edge_start_idx = self.edge_idx_array[idx, 0]
        edge_end_idx = self.edge_idx_array[idx, 1]
        
        # get data pertaining to nodes for this molecule
        positions = self.positions[node_start_idx:node_end_idx]
        atom_types = self.atom_types[node_start_idx:node_end_idx].float()
        atom_charges = self.atom_charges[node_start_idx:node_end_idx].long()

        # remove COM from positions
        positions = positions - positions.mean(dim=0, keepdim=True)
        if self.prior_config.get('x', {}).get('scaling_factor'):
            positions = positions / self.prior_config['x']['scaling_factor']

        # # random rotate the molecular
        # random_matrix = torch.randn(3, 3)
        # Q, R = torch.linalg.qr(random_matrix)
        # d = torch.det(Q)
        # if d < 0:
        #     Q[:, -1] = -Q[:, -1]
        # positions = torch.mm(positions, Q)

        # get data pertaining to edges for this molecule
        bond_types = self.bond_types[edge_start_idx:edge_end_idx].int()
        bond_idxs = self.bond_idxs[edge_start_idx:edge_end_idx].long()


        # reconstruct adjacency matrix
        n_atoms = positions.shape[0]
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)

        # fill in the values of the adjacency matrix specified by bond_idxs
        adj[bond_idxs[:,0], bond_idxs[:,1]] = bond_types

        # get upper triangle of adjacency matrix
        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1) # has shape (2, n_upper_edges)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

        # get lower triangle edges by swapping source and destination of upper_edge_idxs
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        # one-hot encode edge labels and atom charges
        edge_labels = one_hot(edge_labels.to(torch.int64), num_classes=self.n_bond_types).float() # hard-coded assumption of 5 bond types
        try:
            atom_charges = one_hot(atom_charges + 2, num_classes=6).float() # hard-coded assumption that charges are in range [-2, 3]
        except Exception as e:
            print('an atom charge outside of the expected range was encountered')
            print(f'max atom charge: {atom_charges.max()}, min atom charge: {atom_charges.min()}')
            raise e

        # create a dgl graph
        g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

        # add edge features
        g.edata['e_1_true'] = edge_labels

        # add node features
        g.ndata['x_1_true'] = positions
        g.ndata['a_1_true'] = atom_types
        g.ndata['c_1_true'] = atom_charges

        # # 添加样本权重
        # if self.sample_weights is not None:
        #     sample_weight = self.sample_weights[idx]
        #     g.ndata['sample_weights'] = torch.full((n_atoms, 1), sample_weight, dtype=torch.float32)


        # sample prior for node features, coupled to the destination features
        dst_dict = {
            'x': positions,
            'a': atom_types,
            'c': atom_charges
        }
        prior_node_feats = coupled_node_prior(dst_dict=dst_dict, prior_config=self.prior_config)
        for feat in prior_node_feats:
            g.ndata[f'{feat}_0'] = prior_node_feats[feat]

        # sample the prior for the edge features    
        upper_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        n_upper_edges = upper_edge_idxs.shape[1]
        upper_edge_mask[:n_upper_edges] = True
        g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e'])

        return g

class PocketLigandDataset(torch.utils.data.Dataset):
    """
    用于加载蛋白-配体对数据的数据集类。
    每个样本返回一个配体图和一个口袋图。
    """
    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(PocketLigandDataset, self).__init__()

        self.prior_config = prior_config
        self.dataset_config = dataset_config
        self.explicit_aromaticity = dataset_config['explicit_aromaticity']
        self.n_bond_types = 5 if self.explicit_aromaticity else 4
        
        processed_data_dir = Path(dataset_config['processed_data_dir'])
        if not processed_data_dir.exists():
            # Fallback path for repository structure
            processed_data_dir = Path(__file__).parent.parent.parent / processed_data_dir
            if not processed_data_dir.exists():
                raise FileNotFoundError(f"Processed data directory {dataset_config['processed_data_dir']} not found.")
        self.processed_data_dir = processed_data_dir

        # 专门为 'crossdocked' 数据集加载数据
        if dataset_config['dataset_name'] == 'crossdock':
            data_file = processed_data_dir / f'{split}_data_processed.pt'
        else:
            raise NotImplementedError(f"Unsupported dataset_name for PocketLigandDataset: {dataset_config['dataset_name']}")

        data_dict = torch.load(data_file)

        # 加载配体数据
        self.ligand_positions = data_dict['ligand_positions']
        self.ligand_atom_types = data_dict['ligand_atom_types']
        self.ligand_atom_charges = data_dict['ligand_atom_charges']
        self.ligand_bond_types = data_dict['ligand_bond_types']
        self.ligand_bond_idxs = data_dict['ligand_bond_idxs']
        self.ligand_node_idx_array = data_dict['ligand_node_idx_array']
        self.ligand_edge_idx_array = data_dict['ligand_edge_idx_array']

        # 加载口袋数据
        self.pocket_positions = data_dict['pocket_positions']
        self.pocket_atom_types = data_dict['pocket_atom_types']
        self.pocket_node_idx_array = data_dict['pocket_node_idx_array']

    def __len__(self):
        # 数据集的长度由蛋白-配体对的数量决定
        return self.ligand_node_idx_array.shape[0]

    def __getitem__(self, idx):
        # 1. 获取当前样本的索引范围
        # 配体索引
        ligand_node_start = self.ligand_node_idx_array[idx, 0]
        ligand_node_end = self.ligand_node_idx_array[idx, 1]
        ligand_edge_start = self.ligand_edge_idx_array[idx, 0]
        ligand_edge_end = self.ligand_edge_idx_array[idx, 1]
        # 口袋索引
        pocket_node_start = self.pocket_node_idx_array[idx, 0]
        pocket_node_end = self.pocket_node_idx_array[idx, 1]

        # 2. 提取配体数据并构建图 (与MoleculeDataset逻辑相同)
        ligand_pos = self.ligand_positions[ligand_node_start:ligand_node_end]
        ligand_types = self.ligand_atom_types[ligand_node_start:ligand_node_end].float()
        ligand_charges = self.ligand_atom_charges[ligand_node_start:ligand_node_end].long()

        # 3. 提取口袋数据并构建图
        pocket_pos = self.pocket_positions[pocket_node_start:pocket_node_end]
        pocket_types = self.pocket_atom_types[pocket_node_start:pocket_node_end].float()

        ligand_pos = ligand_pos - pocket_pos.mean(dim=0, keepdim=True)
        pocket_pos = pocket_pos - pocket_pos.mean(dim=0, keepdim=True)
        

        if self.prior_config.get('x', {}).get('scaling_factor'):
            pocket_pos = pocket_pos / self.prior_config['x']['scaling_factor']
            ligand_pos = ligand_pos / self.prior_config['x']['scaling_factor']
            
        bond_types = self.ligand_bond_types[ligand_edge_start:ligand_edge_end].int()
        bond_idxs = self.ligand_bond_idxs[ligand_edge_start:ligand_edge_end].long()
        
        n_ligand_atoms = ligand_pos.shape[0]
        adj = torch.zeros((n_ligand_atoms, n_ligand_atoms), dtype=torch.int32)
        # 注意：bond_types 在这里是 one-hot 编码的，需要先转换为类别索引
        adj[bond_idxs[:,0], bond_idxs[:,1]] = bond_types
        
        upper_edge_idxs = torch.triu_indices(n_ligand_atoms, n_ligand_atoms, offset=1)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        edge_labels_one_hot = one_hot(edge_labels.to(torch.int64), num_classes=self.n_bond_types).float()
        ligand_charges_one_hot = one_hot(ligand_charges + 2, num_classes=6).float()

        ligand_g = dgl.graph((edges[0], edges[1]), num_nodes=n_ligand_atoms)
        ligand_g.edata['e_1_true'] = edge_labels_one_hot
        ligand_g.ndata['x_1_true'] = ligand_pos
        ligand_g.ndata['a_1_true'] = ligand_types
        ligand_g.ndata['c_1_true'] = ligand_charges_one_hot

        dst_dict = {'x': ligand_pos, 'a': ligand_types, 'c': ligand_charges_one_hot}
        prior_node_feats = coupled_node_prior(dst_dict=dst_dict, prior_config=self.prior_config)
        for feat in prior_node_feats:
            ligand_g.ndata[f'{feat}_0'] = prior_node_feats[feat]

        upper_edge_mask = torch.zeros(ligand_g.num_edges(), dtype=torch.bool)
        upper_edge_mask[:upper_edge_idxs.shape[1]] = True
        ligand_g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e'])
        
        
        n_pocket_atoms = pocket_pos.shape[0]
        # 创建一个没有边的图来表示口袋
        pocket_g = dgl.graph(([], []), num_nodes=n_pocket_atoms)
        
        # 添加口袋节点特征
        pocket_g.ndata['x_1_true'] = pocket_pos
        pocket_g.ndata['a_1_true'] = pocket_types
        # 口袋没有电荷或先验特征

        # 为口袋建立KNN连接
        pocket_g = self.add_knn_edges(pocket_g)

        return ligand_g, pocket_g

    def add_knn_edges(self, g: dgl.DGLGraph):
        """
        在原图上添加KNN边（支持批处理图，完全在GPU上操作）
        参考reconstruct_graph_dynamic的实现逻辑
        Args:
            g: 输入DGL图（可以是批处理图），将直接修改这个图
        Returns:
            修改后的图
        """
        if g.num_edges() > 0:
            return g  # 如果已有边则不处理
        
        device = g.device
        coords = g.ndata['x_1_true']  # [N, 3]
        num_total_nodes = g.num_nodes()
        
        # 获取批处理信息（模拟node_batch_idx）
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
        node_batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            batch_num_nodes
        )
        
        # 计算每个子图的节点偏移量
        graph_offsets = torch.cat([
            torch.tensor([0], device=device), 
            batch_num_nodes.cumsum(0)[:-1]
        ])
        graph_ranges = torch.stack([graph_offsets, graph_offsets + batch_num_nodes], dim=1)
        
        edges_to_add = torch.tensor([], dtype=torch.long, device=device)
        
        for i in range(batch_size):
            start, end = graph_ranges[i]
            num_nodes = end - start
            if num_nodes < 2:
                continue  # 单节点子图不需要边

            # 获取当前子图的节点坐标
            subgraph_nodes = torch.arange(start, end, device=device)
            subgraph_pos = coords[subgraph_nodes]
            
            # 计算KNN（向量化实现）
            diff = subgraph_pos.unsqueeze(1) - subgraph_pos.unsqueeze(0)  # [N, N, 3]
            dist_matrix = torch.norm(diff, dim=2)  # [N, N]
            
            # 获取每个节点的top k+1最近邻(包括自己)
            k = min(12, num_nodes-1)
            _, topk_indices = torch.topk(dist_matrix, k=k+1, largest=False)
            
            # 生成边对（跳过自身）
            u = torch.repeat_interleave(torch.arange(num_nodes, device=device), k)
            v = topk_indices[:, 1:k+1].flatten()  # 跳过自己(索引0)
            
            # 转换为全局节点索引
            u_global = subgraph_nodes[u]
            v_global = subgraph_nodes[v]
            
            # 创建边ID（双向）
            edge_ids = u_global * num_total_nodes + v_global
            edge_ids_reverse = v_global * num_total_nodes + u_global
            edges_to_add = torch.cat([edges_to_add, edge_ids, edge_ids_reverse])
        
        # 去重并转换为节点对
        edges_to_add = torch.unique(edges_to_add)
        src_ids = edges_to_add // num_total_nodes
        dst_ids = edges_to_add % num_total_nodes
        
        # 添加边到原图（单次操作）
        g.add_edges(src_ids, dst_ids)
        
        # 更新批处理信息
        edge_batch_idx = node_batch_idx[src_ids]
        num_nodes_per_graph = torch.bincount(node_batch_idx, minlength=batch_size)

        g.set_batch_num_nodes(num_nodes_per_graph)
        g.set_batch_num_edges(torch.bincount(edge_batch_idx, minlength=batch_size))
        
        return g    
def collate_fn(batch):
    ligand_gs, pocket_gs = zip(*batch)

    # 分别对 ligand 和 pocket 的图进行批处理
    batched_ligand = dgl.batch(ligand_gs)
    batched_pocket = dgl.batch(pocket_gs)
    
    return batched_ligand, batched_pocket