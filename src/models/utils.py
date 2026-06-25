import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from torch_scatter import segment_csr
from scipy.optimize import linear_sum_assignment
from src.models.gvp import GVP, _rbf, _norm_no_nan
from src.data_processing.geom import MoleculeFeaturizer
##############################################################################################################
# module_utils
##############################################################################################################

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)
    
class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars*2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats


##############################################################################################################
# scheduler_utils
##############################################################################################################
def build_continuous_inv_temp_func(schedule, max_inv_temp=None):

    if schedule is None:
        inv_temp_func = lambda t: 1.0
    elif schedule == 'linear':
        inv_temp_func = lambda t: max_inv_temp*(1 - t)
    elif callable(schedule):
        inv_temp_func = schedule
    else:
        raise ValueError(f'Invalid continuous_inv_temp_schedule: {schedule}')
    return inv_temp_func

def build_cat_temp_schedule(cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a):

    if cat_temperature_schedule == 'decay':
        cat_temp_func = lambda t: cat_temp_decay_max*torch.pow(1-t, cat_temp_decay_a)
    elif isinstance(cat_temperature_schedule, (float, int)):
        cat_temp_func = lambda t: cat_temperature_schedule
    elif callable(cat_temperature_schedule):
        cat_temp_func = cat_temperature_schedule
    else:
        raise ValueError(f"Invalid cat_temperature_schedule: {cat_temperature_schedule}")
    
    return cat_temp_func

def build_fw_schedule(forward_weight_schedule, fw_beta_a, fw_beta_b, fw_beta_max):

    if forward_weight_schedule == 'beta':
        forward_weight_func = lambda t: 1 + fw_beta_max*torch.pow(t, fw_beta_a)*torch.pow(1-t, fw_beta_b)
    elif isinstance(forward_weight_schedule, (float, int)):
        forward_weight_func = lambda t: forward_weight_schedule
    elif callable(forward_weight_schedule):
        forward_weight_func = forward_weight_schedule
    else:
        raise ValueError(f"Invalid forward_weight_schedule: {forward_weight_schedule}")
    
    return forward_weight_func


##############################################################################################################
# algorithm_utils
##############################################################################################################

def compute_ot_permutation(cost_matrix_gpu):

    # 将成本矩阵移至CPU并转换为numpy数组
    cost_matrix_cpu = cost_matrix_gpu.cpu().numpy()
    
    # 使用scipy的linear_sum_assignment求解最优分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)
    
    # 我们需要一个排列p，使得 x_0[p] 最接近 x_1。
    # linear_sum_assignment的结果是 row_ind[i] -> col_ind[i]。
    # 由于row_ind是 [0, 1, 2, ...]，col_ind[i] 就是与第 i 个 x_0 节点匹配的 x_1 节点的索引。
    # 我们希望新的 x_0 (x_0_permuted) 的第 j 个节点，能匹配 x_1 的第 j 个节点。
    # 假设 x_1 的第 j 个节点由 x_0 的第 i 个节点匹配，即 col_ind[i] = j。
    # 那么 x_0_permuted[j] 应该等于 x_0[i]。这需要逆排列。
    permutation = torch.empty_like(torch.from_numpy(col_ind))
    permutation[col_ind] = torch.arange(len(col_ind))
    
    return permutation.to(cost_matrix_gpu.device)

def precompute_distances(g: dgl.DGLGraph, node_positions=None, rbf_dmax = 14, rbf_dim = 32):
    """Precompute the pairwise distances between all nodes in the graph."""

    with g.local_scope():

        if node_positions is None:
            g.ndata['x_d'] = g.ndata['x_t']
        else:
            g.ndata['x_d'] = node_positions

        g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))

        dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
        x_diff = g.edata['x_diff'] / dij
        d = _rbf(dij.squeeze(1), D_max=rbf_dmax, D_count=rbf_dim)
    
    return x_diff, d

def precompute_distances_hetero(g: dgl.DGLGraph, node_positions = None, rbf_dmax = 14, rbf_dim = 32):
    """Precompute the pairwise distances between nodes in a heterogeneous graph.
    
    Args:
        g: 异构图
        node_positions: 可选的节点位置字典 {'ligand': tensor, 'pocket': tensor}
    
    Returns:
        x_diff_dict: 每种边类型的相对位移向量字典
        d_dict: 每种边类型的RBF距离特征字典
    """
    with g.local_scope():
        # 设置节点位置
        if node_positions is None:
            for ntype in g.ntypes:
                if 'x_t' in g.nodes[ntype].data:
                    g.nodes[ntype].data['x_d'] = g.nodes[ntype].data['x_t']
        else:
            for ntype, pos in node_positions.items():
                g.nodes[ntype].data['x_d'] = pos

        x_diff_dict = {}
        d_dict = {}

        # 为每种边类型计算距离
        for etype in g.canonical_etypes:
            if g.num_edges(etype) == 0:
                x_diff_dict[etype] = torch.zeros((0, 3), device=g.device)
                d_dict[etype] = torch.zeros((0, rbf_dim), device=g.device)
                continue
                
            # 计算相对位移
            g.apply_edges(fn.u_sub_v('x_d', 'x_d', 'x_diff'), etype=etype)
            
            # 计算距离并归一化
            dij = _norm_no_nan(g.edges[etype].data['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edges[etype].data['x_diff'] / dij
            
            # 计算RBF距离特征
            d = _rbf(dij.squeeze(1), D_max=rbf_dmax, D_count=rbf_dim)

            x_diff_dict[etype] = x_diff
            d_dict[etype] = d


    return x_diff_dict, d_dict

##############################################################################################################
# intergration_utils
##############################################################################################################

def campbell_step(p_1_given_t: torch.Tensor,
                    xt: torch.Tensor, 
                    stochasticity: float, 
                    hc_thresh: float, 
                    alpha_t: float, 
                    alpha_t_prime: float,
                    dt,
                    batch_size: int,
                    batch_num_nodes: torch.Tensor,
                    n_classes: int,
                    mask_index:int,
                    last_step: bool, 
                    batch_idx: torch.Tensor,
                ):
    x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)

    mask_prob = dt*stochasticity
    unmask_prob = dt*( alpha_t_prime + stochasticity*alpha_t  ) / (1 - alpha_t)
    # unmask_prob = alpha_t + mask_prob - (xt == mask_index).float().mean().item()
    
    mask_prob = torch.clamp(mask_prob, min=0, max=1)
    unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
    

    # sample which nodes will be unmasked
    if hc_thresh > 0:
        # select more high-confidence predictions for unmasking than low-confidence predictions
        will_unmask = purity_sampling(
            xt=xt, x1=x1, x1_probs=p_1_given_t, unmask_prob=unmask_prob,
            mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
            node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
    else:
        # uniformly sample nodes to unmask
        will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
        will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked

    if not last_step:
        # compute which nodes will be masked
        will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
        will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

        # mask the nodes
        xt[will_mask] = mask_index

    # unmask the nodes
    xt[will_unmask] = x1[will_unmask]

    xt = one_hot(xt, num_classes=n_classes).float()
    x1 = one_hot(x1, num_classes=n_classes).float()
    return xt, x1


def campbell_step_pocket(p_1_given_t: torch.Tensor,
                    p_1_given_t_no_pocket: torch.Tensor,
                    xt: torch.Tensor, 
                    stochasticity: float, 
                    hc_thresh: float, 
                    alpha_t: float, 
                    alpha_t_prime: float,
                    dt,
                    batch_size: int,
                    batch_num_nodes: torch.Tensor,
                    n_classes: int,
                    mask_index:int,
                    last_step: bool, 
                    batch_idx: torch.Tensor,
                ):

    x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)
    x1_no_pocket = Categorical(p_1_given_t_no_pocket).sample() # has shape (num_nodes,)

    mask_prob = dt*stochasticity
    unmask_prob = 1.5*dt*( alpha_t_prime + stochasticity*alpha_t  ) / (1 - alpha_t)
    # unmask_prob = alpha_t + mask_prob - (xt != mask_index).float().mean().item()
    
    mask_prob = torch.clamp(mask_prob, min=0, max=1)
    unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
    

    # sample which nodes will be unmasked
    if hc_thresh > 0:
        # select more high-confidence predictions for unmasking than low-confidence predictions
        will_unmask = purity_sampling_pocket(
            xt=xt, x1=x1, x1_probs=p_1_given_t, x1_probs_no_pocket=p_1_given_t_no_pocket, unmask_prob=unmask_prob,
            mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
            node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
        will_unmask = will_unmask * (x1 == x1_no_pocket)
    else:
        # uniformly sample nodes to unmask
        will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
        will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked
        
    if not last_step:
        will_unmask = will_unmask * (x1 == x1_no_pocket) 

    if not last_step:
        # compute which nodes will be masked
        will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
        will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

        # mask the nodes
        xt[will_mask] = mask_index

    # unmask the nodes
    xt[will_unmask] = x1[will_unmask]

    xt = one_hot(xt, num_classes=n_classes).float()
    x1 = one_hot(x1, num_classes=n_classes).float()
    return xt, x1

def gat_step(
            p_1_given_t: torch.Tensor,
            xt: torch.Tensor, 
            alpha_t: float, 
            alpha_t_prime: float,
            forward_weight: float,
            dt,
            batch_size: int,
            batch_num_nodes: torch.Tensor,
            n_classes: int,
            mask_index:int,
            batch_idx: torch.Tensor,
        ):


    # add a zero-column on to p_1_given_t to represent the mask token
    p_1_given_t = torch.cat([p_1_given_t, torch.zeros_like(p_1_given_t[:, :1])], dim=-1)

    # create a one-hot encoding of xt
    delta_xt = one_hot(xt, num_classes=n_classes).float()

    # compute forward probability velocity
    u_forward = alpha_t_prime / (1 - alpha_t) * (p_1_given_t - delta_xt)

    # create a delta on the mask token
    delta_mask = torch.zeros_like(delta_xt)
    delta_mask[:, mask_index] = 1

    # compute the backward probability velocity
    u_backward = alpha_t_prime / (alpha_t + 1e-8) * (delta_xt - delta_mask)

    # compute the probability velocity
    backward_weight = forward_weight - 1
    pvel = forward_weight*u_forward - backward_weight*u_backward

    # compute the parameters of the transition distritibution
    p_step = delta_xt + dt*pvel

    # clamp p_step to be valid
    p_step = torch.clamp(p_step, min=1.0e-9, max=1)

    # sample x_{t+dt} from the transition distribution
    x_dt = Categorical(p_step).sample()

    # one-hot encode x_{t+dt}
    x_dt = one_hot(x_dt, num_classes=n_classes).float()

    return x_dt

def purity_sampling(xt, x1, x1_probs, unmask_prob, mask_index, batch_size, batch_num_nodes, node_batch_idx, hc_thresh, device):

    masked_nodes = xt == mask_index # mask of which nodes are currently unmasked
    purities = x1_probs.max(-1)[0] # the highest probability of any category for each node

    hc_mask = purities >= hc_thresh # mask of which nodes are high-confidence
    hc_mask = hc_mask * masked_nodes # only consider nodes that are currently masked

    # compute the number of hc nodes in each graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    hc_nodes_per_graph = segment_csr(hc_mask.long(), indptr) # has shape (batch_size,)

    # compute the number of masked nodes in each graph in the batch
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)

    # compute max value of ph for each graph in the batch
    ph_max = unmask_prob*masked_nodes_per_graph / hc_nodes_per_graph
    ph_max[ hc_nodes_per_graph == 0 ] = torch.inf

    # compute ph and pl for each graph in the batch
    ph = torch.minimum(ph_max, torch.full_like(ph_max, 1.0)) # bernoulli trial probability of high confidence nodes in each graph
    pl = (unmask_prob*masked_nodes_per_graph - ph*hc_nodes_per_graph) / (masked_nodes_per_graph - hc_nodes_per_graph) # bernoulli trial probability of low confidence nodes in each graph

    # construct a tensor containing the unmask probability for every node
    node_unmask_prob = torch.zeros_like(xt).float()
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes # nodes which are currently masked and low-confidence
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    will_unmask = torch.rand(xt.shape[0], device=device) < node_unmask_prob # sample nodes to unmask
    return will_unmask

def purity_sampling_pocket(xt, x1, x1_probs, x1_probs_no_pocket, unmask_prob, mask_index, batch_size, batch_num_nodes, node_batch_idx, hc_thresh, device):

    masked_nodes = xt == mask_index
    # 计算联合置信度 - 两种条件预测的乘积
    joint_probs = x1_probs * x1_probs_no_pocket  # 逐元素乘积
    purities = joint_probs.max(-1)[0]    # 取最大联合概率作为置信度
    
    hc_mask = purities >= hc_thresh
    hc_mask = hc_mask * masked_nodes

    # compute the number of hc nodes in each graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    hc_nodes_per_graph = segment_csr(hc_mask.long(), indptr) # has shape (batch_size,)

    # compute the number of masked nodes in each graph in the batch
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)

    # compute max value of ph for each graph in the batch
    ph_max = unmask_prob*masked_nodes_per_graph / hc_nodes_per_graph
    ph_max[ hc_nodes_per_graph == 0 ] = torch.inf

    # compute ph and pl for each graph in the batch
    ph = torch.minimum(ph_max, torch.full_like(ph_max, 1.0)) # bernoulli trial probability of high confidence nodes in each graph
    pl = (unmask_prob*masked_nodes_per_graph - ph*hc_nodes_per_graph) / (masked_nodes_per_graph - hc_nodes_per_graph) # bernoulli trial probability of low confidence nodes in each graph

    # construct a tensor containing the unmask probability for every node
    node_unmask_prob = torch.zeros_like(xt).float()
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes # nodes which are currently masked and low-confidence
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    will_unmask = torch.rand(xt.shape[0], device=device) < node_unmask_prob # sample nodes to unmask
    return will_unmask
##############################################################################################################
# graph_utils
##############################################################################################################

def reconstruct_graph_dynamic(
        g: dgl.DGLGraph,
        upper_edge_mask: torch.Tensor,
        node_batch_idx: torch.Tensor,
        k: int = 12,
        # cutoff: float = 6.0,
        null_edge_value: float = 1.0,
        device: torch.device = None
    ):
    if device is None:
        device = g.device

    original_batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
    num_total_nodes = g.num_nodes()
    assert num_total_nodes == len(node_batch_idx), "node_batch_idx长度与节点数不匹配"
    assert g.num_edges() == len(upper_edge_mask), "upper_edge_mask长度与边数不匹配"

    # 1. 构建边ID的索引
    src, dst = g.edges()
    edges_initial = src * num_total_nodes + dst

    # 获取节点坐标
    node_positions = g.ndata['x_t'][:, :3]

    # 2. 识别需要保留的边(real边)


    # 3. 计算每个子图需要多少边
    batch_size = original_batch_size
    num_nodes_per_graph = torch.bincount(node_batch_idx, minlength=batch_size)

    
    # 4. 为每个子图采样需要添加的边ID
    graph_offsets = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(0)[:-1]])
    graph_ranges = torch.stack([graph_offsets, graph_offsets + num_nodes_per_graph], dim=1)
    
    edges_to_keep = torch.tensor([], dtype=torch.long, device=device)
    for i in range(batch_size):
        
        start, end = graph_ranges[i]
        num_nodes = end - start
        if num_nodes < 2:
            continue

        # 获取当前子图的节点坐标
        subgraph_nodes = torch.arange(start, end, device=device)
        subgraph_pos = node_positions[subgraph_nodes]
        
        # 计算KNN
        # 使用距离矩阵实现KNN
        diff = subgraph_pos.unsqueeze(1) - subgraph_pos.unsqueeze(0)  # [N, N, 3]
        dist_matrix = torch.norm(diff, dim=2)  # [N, N]
        
        # 获取每个节点的top k+1最近邻(包括自己)
        k = min(k, num_nodes-1)
        topk_values, topk_indices = torch.topk(dist_matrix, k=k+1, largest=False, dim=1)
        
        # 生成边对
        u = torch.repeat_interleave(torch.arange(num_nodes, device=device), k)
        v = topk_indices[:, 1:k+1].flatten()  # 跳过自己(索引0)

        distances = dist_matrix[u, v]
    
        # # 创建掩码：只保留距离小于截断值的边
        # distance_mask = distances < cutoff
        
        # # 应用距离截断
        # u = u[distance_mask]
        # v = v[distance_mask]
        
        # 转换为全局节点索引
        u_global = subgraph_nodes[u]
        v_global = subgraph_nodes[v]
        
        # 创建双向边
        edge_ids = u_global * num_total_nodes + v_global
        edge_ids_reverse = v_global * num_total_nodes + u_global
        
        # 添加到边列表
        edges_to_keep = torch.cat([edges_to_keep, edge_ids, edge_ids_reverse])
    
    # 去重
    edges_to_keep = torch.unique(edges_to_keep)
    edges_to_add = edges_to_keep[~torch.isin(edges_to_keep, edges_initial)]
    edges_to_remove = edges_initial[~torch.isin(edges_initial, edges_to_keep)]

    # 7. 执行边的删除和添加操作
    # 先删除需要删除的边
    if len(edges_to_remove) > 0:
        src, dst = g.edges()
        current_edge_ids = src * num_total_nodes + dst
        remove_mask = torch.isin(current_edge_ids, edges_to_remove)
        eids_to_remove = remove_mask.nonzero().flatten()
        g.remove_edges(eids_to_remove)


    # 添加新边
    if len(edges_to_add) > 0:
        src_ids = edges_to_add // num_total_nodes
        dst_ids = edges_to_add % num_total_nodes
        
        # 保存旧特征（避免多次访问edata）
        old_feats = {name: g.edata[name] for name in g.edata.keys()}

        
        # 批量添加边（单次操作比多次添加高效）
        g.add_edges(src_ids, dst_ids)
        
        # 设置所有特征
        for feat_name, old_feat in old_feats.items():
            feat_dim = old_feat.shape[-1] if len(old_feat.shape) > 1 else 1
            dtype = old_feat.dtype
            
            # 预分配内存（直接创建最终大小的张量）
            new_feat = torch.zeros((g.num_edges(), *old_feat.shape[1:]), 
                                dtype=dtype, device=device)
            
            # 保留旧特征（向量化复制）
            num_old_edges = old_feat.shape[0]
            new_feat[:num_old_edges] = old_feat
            
            # 批量设置新边特征（避免循环）
            if feat_name == 'e_t':
                # 特殊处理e_t特征
                new_feat[num_old_edges:, -1] = null_edge_value
            else:
                # 其他特征保持默认零值
                pass
                
            g.edata[feat_name] = new_feat
    
    # 8. 更新upper_edge_mask和edge_batch_idx
    src, dst = g.edges()

    edge_batch_idx = node_batch_idx[src]
    upper_edge_mask = src < dst

    edge_ids = torch.min(src, dst) * num_total_nodes + torch.max(src, dst)

    sort_keys = (edge_batch_idx * 2 + (~upper_edge_mask).long()) * (num_total_nodes * num_total_nodes) + edge_ids

    # 9. 按edge_ids排序
    sorted_indices = torch.argsort(sort_keys)
    g = dgl.reorder_graph(
        g,
        node_permute_algo=None,
        edge_permute_algo='custom',
        permute_config={'edges_perm': sorted_indices},
        store_ids=True
    )

    src, dst = g.edges()
    edge_batch_idx_new = node_batch_idx[src]
    upper_edge_mask_new = src < dst


    # 更新图的批处理信息
    g.set_batch_num_nodes(num_nodes_per_graph)
    g.set_batch_num_edges(torch.bincount(edge_batch_idx_new, minlength=batch_size))


    return g, upper_edge_mask_new, edge_batch_idx_new

def reconstruct_graph_with_frag(
        g: dgl.DGLGraph,
        upper_edge_mask: torch.Tensor,
        node_batch_idx: torch.Tensor,
        frag_features: dict,
        k: int = 12,
        null_edge_value: float = 1.0,
        t: float = 0.0,
        dt: float = 0.004,
        device: torch.device = None
    ):
    """
    从片段特征重建图结构，将片段信息应用到每个子图的前num_atoms个节点
    
    Args:
        g: 输入的DGL图
        upper_edge_mask: 上三角边掩码
        node_batch_idx: 节点批次索引
        frag_features: 片段特征字典 {'x': positions, 'a': atom_types, 'c': atom_charges, 'e': bond_types, 'e_pair': bond_pairs}
        k: KNN参数
        null_edge_value: 空边特征值
        device: 设备
    
    Returns:
        重建后的图，更新后的upper_edge_mask和edge_batch_idx
    """
    if device is None:
        device = g.device

    # 解包片段特征
    template_positions = frag_features['x'][0].float().to(device) - frag_features['x'][0].float().mean(dim=0).to(device)
    template_atom_types =  torch.cat([frag_features['a'][0].float().to(device), torch.zeros(frag_features['a'][0].shape[0], 1, device=device)], dim=-1)
    template_atom_charges = one_hot(frag_features['c'][0].long() + 2, num_classes=g.ndata['c_t'].shape[1]).float().to(device)
    # template_bond_types = one_hot(frag_features['e'][0].long(), num_classes=g.edata['e_t'].shape[1]).float().to(device)
    template_bond_types = frag_features['e'][0].int()
    template_bond_pairs = frag_features['e_pair'][0].to(device)


    # 获取边的类别数
    num_bond_classes = g.edata['e_t'].shape[1]

    # 重建邻接矩阵
    n_atoms = template_positions.shape[0]
    adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)

    # 使用模板边对填充邻接矩阵，所有边类型设为0
    adj[template_bond_pairs[:, 0], template_bond_pairs[:, 1]] = template_bond_types

    # 获取上三角部分的边索引
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
    upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

    # 获取下三角部分（通过交换源和目标节点）
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    # 合并上下三角的边
    edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

    # one-hot编码边标签
    template_bond_types = one_hot(edge_labels.to(torch.int64), num_classes=num_bond_classes).float().to(device)
    template_bond_pairs = edges.t().to(device)  # 转置为 (n_edges, 2) 格式



    num_atoms = template_positions.shape[0]
    
    
    original_batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
    num_total_nodes = g.num_nodes()
    assert num_total_nodes == len(node_batch_idx), "node_batch_idx长度与节点数不匹配"
    assert g.num_edges() == len(upper_edge_mask), "upper_edge_mask长度与边数不匹配"

    # 1. 计算每个子图的节点范围
    batch_size = original_batch_size
    num_nodes_per_graph = torch.bincount(node_batch_idx, minlength=batch_size)
    graph_offsets = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(0)[:-1]])
    graph_ranges = torch.stack([graph_offsets, graph_offsets + num_nodes_per_graph], dim=1)
    
    # 检查每个子图是否有足够的节点容纳片段
    num_nodes_per_subgraph = graph_ranges[:, 1] - graph_ranges[:, 0]
    if torch.any(num_nodes_per_subgraph < num_atoms):
        raise ValueError("某些子图没有足够的节点容纳片段")


    # global_permutation_list = []
    # with torch.no_grad():
    #     for i in range(batch_size):
    #         # 提取当前子图的节点和它们的初始位置 x_0
    #         start_node_idx = graph_offsets[i]
    #         num_subgraph_nodes = num_nodes_per_graph[i]
    #         end_node_idx = start_node_idx + num_subgraph_nodes
            
    #         subgraph_x0 = g.ndata['x_0'][start_node_idx:end_node_idx]

    #         # 计算成本矩阵：片段原子与子图节点之间的距离
    #         # cost_matrix shape: (num_atoms, num_subgraph_nodes)
    #         cost_matrix = torch.cdist(template_positions, subgraph_x0)
            
    #         # 使用scipy求解线性分配问题 (需要移到CPU)
    #         cost_matrix_cpu = cost_matrix.cpu().numpy()
    #         frag_indices, subgraph_local_indices = linear_sum_assignment(cost_matrix_cpu)
            
    #         # subgraph_local_indices 是在子图内部的索引，它们是与片段原子最佳匹配的节点
    #         matched_nodes_local = torch.from_numpy(subgraph_local_indices).to(device).long()
            
    #         # 创建该子图的节点排列
    #         # 首先是匹配上的节点
    #         perm_local = torch.zeros(num_subgraph_nodes, dtype=torch.long, device=device)
    #         perm_local[:num_atoms] = matched_nodes_local
            
    #         # 然后是未匹配的节点
    #         is_matched_mask = torch.zeros(num_subgraph_nodes, dtype=torch.bool, device=device)
    #         is_matched_mask[matched_nodes_local] = True
    #         unmatched_nodes_local = torch.arange(num_subgraph_nodes, device=device)[~is_matched_mask]
    #         perm_local[num_atoms:] = unmatched_nodes_local
            
    #         # 将局部排列转换为全局索引
    #         perm_global_for_subgraph = start_node_idx + perm_local
    #         global_permutation_list.append(perm_global_for_subgraph)

    # # --- 4. 重构图及相关张量 ---
    # # 将所有子图的排列拼接成一个全局排列
    # global_permutation = torch.cat(global_permutation_list)
    
    # # 使用DGL的reorder_graph函数来重排图的节点
    # # node_perm[new_idx] = old_idx
    # # 这意味着新图的第i个节点将是旧图的第 global_permutation[i] 个节点
    # g = dgl.reorder_graph(g, node_permute_algo='custom', permute_config={'nodes_perm': global_permutation}, store_ids=False)
    
    # # 更新与节点和边顺序相关的张量
    # node_batch_idx = node_batch_idx[global_permutation]
    # src, dst = g.edges()
    # upper_edge_mask = src < dst
    # edge_batch_idx = node_batch_idx[src]

    # edge_ids = torch.min(src, dst) * num_total_nodes + torch.max(src, dst)
    # sort_keys = (edge_batch_idx * 2 + (~upper_edge_mask).long()) * (num_total_nodes * num_total_nodes) + edge_ids

    # sorted_indices = torch.argsort(sort_keys)
    # g = dgl.reorder_graph(
    #     g,
    #     node_permute_algo=None,
    #     edge_permute_algo='custom',
    #     permute_config={'edges_perm': sorted_indices},
    #     store_ids=True
    # )
    

    # # 此时，g 中每个子图的前 num_atoms 个节点就是OT匹配到的节点
    # # 重新计算子图的节点范围（尽管在这个新图中它们将是连续的，但为了代码清晰）
    # num_nodes_per_graph = torch.bincount(node_batch_idx, minlength=batch_size)
    # upper_edge_mask = src < dst
    # edge_batch_idx = node_batch_idx[src]

    # g.set_batch_num_nodes(num_nodes_per_graph)
    # g.set_batch_num_edges(torch.bincount(edge_batch_idx, minlength=batch_size))


    # 2. 批量更新节点特征
    with torch.no_grad():
        # 创建片段节点掩码
        frag_node_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=device)
        frag_node_indices = []
        
        for i in range(batch_size):
            start = graph_ranges[i, 0]
            frag_start = start
            frag_end = start + num_atoms
            frag_node_mask[frag_start:frag_end] = True
            frag_node_indices.append(torch.arange(frag_start, frag_end, device=device))
        
        frag_node_indices = torch.cat(frag_node_indices)

        # 批量更新节点坐标
        # x_{t+1}_corrected = x_{t+1}^unconditional + C(t) * [M ⊙ (M ⊙ x̂₁ - y)] * Δt

        g.ndata['x_t'][frag_node_indices, :3] =  g.ndata['x_t'][frag_node_indices, :3] + (1-t-dt)/(t+dt) * (template_positions.repeat(batch_size, 1) - g.ndata['x_1_pred'][frag_node_indices, :3]) * dt
        

        
        # 批量更新原子类型
        if 'a_t' in g.ndata:
            remask_indices = torch.any(g.ndata['a_t'][frag_node_indices] != template_atom_types.repeat(batch_size, 1), dim=1)
            g.ndata['a_t'][frag_node_indices[remask_indices]] = torch.zeros(g.ndata['a_t'][frag_node_indices[remask_indices]].shape[0], g.ndata['a_t'].shape[1], device=device)
            g.ndata['a_t'][frag_node_indices[remask_indices], -1] = 1

        
        # 批量更新电荷信息
        if 'c_t' in g.ndata:
            remask_indices = torch.any(g.ndata['c_t'][frag_node_indices] != template_atom_charges.repeat(batch_size, 1), dim=1)
            g.ndata['c_t'][frag_node_indices[remask_indices]] = torch.zeros(g.ndata['c_t'][frag_node_indices[remask_indices]].shape[0], g.ndata['c_t'].shape[1], device=device)
            g.ndata['c_t'][frag_node_indices[remask_indices], -1] = 1



    # 3. 预计算所有KNN边（向量化）
    # 构建所有子图的KNN边
    all_knn_src = []
    all_knn_dst = []
    
    for i in range(batch_size):
        
        start, end = graph_ranges[i]
        num_nodes = end - start
        if num_nodes < 2:
            continue

        # 获取当前子图的节点坐标
        subgraph_nodes = torch.arange(start, end, device=device)
        subgraph_pos = g.ndata['x_t'][subgraph_nodes, :3]
        
        # 计算KNN
        # 使用距离矩阵实现KNN
        diff = subgraph_pos.unsqueeze(1) - subgraph_pos.unsqueeze(0)  # [N, N, 3]
        dist_matrix = torch.norm(diff, dim=2)  # [N, N]
        
        # 获取每个节点的top k+1最近邻(包括自己)
        k = min(k, num_nodes-1)
        topk_values, topk_indices = torch.topk(dist_matrix, k=k+1, largest=False, dim=1)
        
        # 生成边对
        u = torch.repeat_interleave(torch.arange(num_nodes, device=device), k)
        v = topk_indices[:, 1:k+1].flatten()  # 跳过自己(索引0)

        distances = dist_matrix[u, v]
    
        # # 创建掩码：只保留距离小于截断值的边
        # distance_mask = distances < cutoff
        
        # # 应用距离截断
        # u = u[distance_mask]
        # v = v[distance_mask]
        
        # 转换为全局节点索引
        u_global = subgraph_nodes[u]
        v_global = subgraph_nodes[v]
        
        # 添加双向边
        all_knn_src.append(torch.cat([u_global, v_global]))
        all_knn_dst.append(torch.cat([v_global, u_global]))
    
    # 4. 预计算所有片段键边（向量化）
    # 构建片段键边映射
    frag_bond_src_local = template_bond_pairs[:, 0]
    frag_bond_dst_local = template_bond_pairs[:, 1]
    frag_bond_features = template_bond_types
    
    # 为每个子图生成片段键边
    all_frag_bond_src = []
    all_frag_bond_dst = []

    for i in range(batch_size):
        start = graph_ranges[i, 0]
        frag_start = start
        
        # 转换为全局索引
        frag_bond_src_global = frag_start + frag_bond_src_local
        frag_bond_dst_global = frag_start + frag_bond_dst_local
        
        # 添加双向边
        all_frag_bond_src.append(frag_bond_src_global)
        all_frag_bond_dst.append(frag_bond_dst_global)

    # 5. 合并所有边并去重
    # 合并KNN边
    if all_knn_src:
        knn_src = torch.cat(all_knn_src)
        knn_dst = torch.cat(all_knn_dst)
    else:
        knn_src = torch.tensor([], device=device, dtype=torch.long)
        knn_dst = torch.tensor([], device=device, dtype=torch.long)
    
    # 合并片段键边
    if all_frag_bond_src:
        frag_bond_src = torch.cat(all_frag_bond_src).to(device)
        frag_bond_dst = torch.cat(all_frag_bond_dst).to(device)
    else:
        frag_bond_src = torch.tensor([], device=device, dtype=torch.long)
        frag_bond_dst = torch.tensor([], device=device, dtype=torch.long)
    
    # 合并所有新边
    all_new_src = torch.cat([knn_src, frag_bond_src])
    all_new_dst = torch.cat([knn_dst, frag_bond_dst])
    
    # 创建唯一边标识并去重
    new_edge_ids = all_new_src * num_total_nodes + all_new_dst
    edges_to_keep = torch.unique(new_edge_ids)

    src, dst = g.edges()
    edges_initial = src * num_total_nodes + dst

    edges_to_add = edges_to_keep[~torch.isin(edges_to_keep, edges_initial)]
    edges_to_remove = edges_initial[~torch.isin(edges_initial, edges_to_keep)]


    # 7. 一次性执行图操作
    # 删除旧边
    if len(edges_to_remove) > 0:
        src, dst = g.edges()
        current_edge_ids = src * num_total_nodes + dst
        remove_mask = torch.isin(current_edge_ids, edges_to_remove)
        eids_to_remove = remove_mask.nonzero().flatten()
        g.remove_edges(eids_to_remove)

    # 添加新边
    if len(edges_to_add) > 0:
        src_ids = edges_to_add // num_total_nodes
        dst_ids = edges_to_add % num_total_nodes
        
        # 保存旧特征（避免多次访问edata）
        old_feats = {name: g.edata[name] for name in g.edata.keys()}

        
        # 批量添加边（单次操作比多次添加高效）
        g.add_edges(src_ids, dst_ids)
        
        # 设置所有特征
        for feat_name, old_feat in old_feats.items():
            feat_dim = old_feat.shape[-1] if len(old_feat.shape) > 1 else 1
            dtype = old_feat.dtype
            
            # 预分配内存（直接创建最终大小的张量）
            new_feat = torch.zeros((g.num_edges(), *old_feat.shape[1:]), 
                                dtype=dtype, device=device)
            
            # 保留旧特征（向量化复制）
            num_old_edges = old_feat.shape[0]
            new_feat[:num_old_edges] = old_feat
            
            # 批量设置新边特征（避免循环）
            if feat_name == 'e_t':
                # 特殊处理e_t特征
                new_feat[num_old_edges:, -1] = null_edge_value
            else:
                # 其他特征保持默认零值
                pass
                
            g.edata[feat_name] = new_feat
    
    if 'e_t' in g.edata and len(frag_bond_src) > 0:
        with torch.no_grad():
            # 构建片段键边的全局ID
            frag_edge_ids = frag_bond_src * num_total_nodes + frag_bond_dst
            
            # 获取图中所有边的全局ID
            src_all, dst_all = g.edges()
            upper_edge_mask = src_all < dst_all
            edge_ids_all = src_all * num_total_nodes + dst_all
            
            # 找到匹配的边索引
            frag_edge_indices = torch.where(torch.isin(edge_ids_all, frag_edge_ids))[0]
            
            if len(frag_edge_indices) > 0:
                remask_indices = torch.any(g.edata['e_t'][frag_edge_indices] != frag_bond_features.repeat(batch_size, 1), dim=1)
                g.edata['e_t'][frag_edge_indices[remask_indices]] = torch.zeros(g.edata['e_t'][frag_edge_indices[remask_indices]].shape[0], g.edata['e_t'].shape[1], device=device)
                g.edata['e_t'][frag_edge_indices[remask_indices], -1] = 1


    # 8. 重新排序图
    src, dst = g.edges()
    edge_batch_idx_new = node_batch_idx[src]
    upper_edge_mask_new = src < dst

    edge_ids = torch.min(src, dst) * num_total_nodes + torch.max(src, dst)
    sort_keys = (edge_batch_idx_new * 2 + (~upper_edge_mask_new).long()) * (num_total_nodes * num_total_nodes) + edge_ids

    sorted_indices = torch.argsort(sort_keys)
    g = dgl.reorder_graph(
        g,
        node_permute_algo=None,
        edge_permute_algo='custom',
        permute_config={'edges_perm': sorted_indices},
        store_ids=True
    )

    # 9. 更新批处理信息
    src, dst = g.edges()
    edge_batch_idx_new = node_batch_idx[src]
    upper_edge_mask_new = src < dst

    g.set_batch_num_nodes(num_nodes_per_graph)
    g.set_batch_num_edges(torch.bincount(edge_batch_idx_new, minlength=batch_size))

    return g, upper_edge_mask_new, edge_batch_idx_new