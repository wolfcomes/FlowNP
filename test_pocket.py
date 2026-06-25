import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from src.models.flowmol import PocketFlowMol
from src.analysis.molecule_builder import SampledMolecule
# from src.analysis.metrics import SampleAnalyzer
from typing import List, Dict, Tuple
from rdkit import Chem
from src.model_utils.load import read_config_file
import pickle
import math
import time
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Geometry import Point3D
import dgl
import os
import numpy as np
from collections import defaultdict

def validate_molecules(molecules: List[SampledMolecule]) -> Dict[str, List]:
    """
    完全模拟 SDMolSupplier 的验证标准，并包含断点和游离原子验证
    """
    valid_mols = []
    invalid_mols = []
    
    # 临时保存分子到SDF
    temp_file = "temp_validation.sdf"
    
    # 写入临时SDF文件
    writer = Chem.SDWriter(temp_file)
    for mol in molecules:
        if mol.rdkit_mol is not None:
            try:
                writer.write(mol.rdkit_mol)
            except:
                pass
    writer.close()
    
    # 用SDMolSupplier读取验证
    supplier = Chem.SDMolSupplier(temp_file)
    supplier_mols = []  # 存储SDMolSupplier成功读取的分子
    
    # 第一轮验证：SDMolSupplier 基础验证
    for rdkit_mol in supplier:
        supplier_mols.append(rdkit_mol)
    
    # 第二轮验证：对SDMolSupplier返回的分子进行断点和游离原子验证
    valid_supplier_indices = set()
    for i, rdkit_mol in enumerate(supplier_mols):
        if rdkit_mol is not None:
            try:
                # 进行SMILES转换
                smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
                
                # 检查断点和游离原子
                has_fragments = False
                # 检查是否有未连接的原子（如 "." 分隔的片段）
                if "." in smiles:
                    has_fragments = True
                else:
                    # 检查分子中是否有未连接的原子（如游离原子）
                    for atom in rdkit_mol.GetAtoms():
                        if atom.GetDegree() == 0 and atom.GetAtomicNum() != 0:
                            has_fragments = True
                            break
                
                if not has_fragments:
                    valid_supplier_indices.add(i)
            except:
                # 如果SMILES转换或验证失败，跳过该分子
                pass
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # 映射回原始分子
    valid_count = 0
    for i, mol in enumerate(molecules):
        if mol.rdkit_mol is None:
            invalid_mols.append(mol)
            continue
            
        # 检查是否在有效的supplier分子中
        if valid_count in valid_supplier_indices:
            valid_mols.append(mol)
        else:
            invalid_mols.append(mol)
        
        valid_count += 1
    
    return {
        'valid': valid_mols,
        'invalid': invalid_mols
    }

def add_knn_edges(g: dgl.DGLGraph):
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

def process_protein_input(pdb_input, atom_map, n_molecules_per_protein=1, ref_ligands_dir=None) -> Tuple[Dict[str, List[dgl.DGLGraph]], Dict[str, np.ndarray]]:
    """
    处理蛋白质输入，返回DGL图字典和口袋/配体质心字典。
    
    参数:
        pdb_input: str或Path - PDB文件路径或包含PDB文件的目录
        atom_map: list - 原子类型映射列表
        n_molecules_per_protein: int - 每个蛋白质生成的图数量
        ref_ligands_dir: Path或None - 参考配体目录，如果提供则使用配体质心而不是口袋质心
        
    返回:
        Tuple[Dict, Dict] - 第一个字典将蛋白质基名映射到DGL图列表，
                           第二个字典将蛋白质基名映射到其口袋或配体的质心(COM)坐标。
    """
    if os.path.isdir(pdb_input):
        pdb_files = [Path(pdb_input) / f for f in os.listdir(pdb_input) 
                     if f.endswith('.pdb') or f.endswith('.ent')]
    elif os.path.isfile(pdb_input) and (str(pdb_input).endswith('.pdb') or str(pdb_input).endswith('.ent')):
        pdb_files = [Path(pdb_input)]
    else:
        raise ValueError("输入必须是PDB文件或包含PDB文件的目录")
    
    protein_graph_map = {}
    protein_com_map = {}  # 用于存储每个口袋或配体的质心
    pdb_parser = PDBParser(QUIET=True)
    
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL', 'NAP'
    }
    
    # 如果有参考配体目录，预先加载所有配体文件
    ligand_files_map = {}
    if ref_ligands_dir is not None and os.path.isdir(ref_ligands_dir):
        ligand_files = [f for f in os.listdir(ref_ligands_dir) if f.endswith('.sdf')]
        for ligand_file in ligand_files:
            # 提取配体文件前缀（最后一个'_'之前的部分）
            ligand_prefix = ligand_file.split('.')[0]
            ligand_files_map[ligand_prefix] = Path(ref_ligands_dir) / ligand_file
    
    for pdb_file in pdb_files:
        protein_name = pdb_file.stem
        try:
            structure = pdb_parser.get_structure(protein_name, pdb_file)
            
            pocket_coords = []
            pocket_atom_symbols = []
            
            for residue in structure.get_residues():
                resname = residue.get_resname().strip()
                if resname == 'HOH' or resname not in standard_residues:
                    continue
                for atom in residue.get_atoms():
                    atom_symbol = atom.element.strip().capitalize()
                    if atom_symbol in atom_map and atom_symbol != 'H':
                        pocket_coords.append(atom.get_coord())
                        pocket_atom_symbols.append(atom_symbol)
            
            if not pocket_coords:
                print(f"警告: 在 {pdb_file} 中没有找到有效的蛋白质原子，已跳过。")
                continue
            
            pocket_coords = np.array(pocket_coords)
            
            # --- 计算质心(COM) ---
            if ref_ligands_dir is not None:
                # 使用配体质心
                # 提取蛋白质名前缀（最后一个'_'之前的部分）
                protein_prefix = protein_name.rsplit('_', 1)[0]
                ligand_file = ligand_files_map.get(protein_prefix)

                if ligand_file is not None and os.path.exists(ligand_file):
                    try:
                        # 读取配体文件并计算质心
                        supplier = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
                        ligand_mol = None
                        for mol in supplier:
                            if mol is not None:
                                ligand_mol = mol
                                break
                        print(str(ligand_file))
                        print(ligand_mol)
                        if ligand_mol is not None and ligand_mol.GetNumAtoms() > 0:
                            # 计算配体质心
                            conf = ligand_mol.GetConformer()
                            ligand_coords = []
                            for i in range(ligand_mol.GetNumAtoms()):
                                pos = conf.GetAtomPosition(i)
                                ligand_coords.append([pos.x, pos.y, pos.z])
                            
                            ligand_coords = np.array(ligand_coords)
                            pocket_com = ligand_coords.mean(axis=0)
                            print(f"使用配体质心: {protein_name} -> {ligand_file.name}")
                        else:
                            # 如果配体文件无效，使用口袋质心
                            pocket_com = pocket_coords.mean(axis=0)
                            print(f"警告: 配体文件 {ligand_file} 无效，使用口袋质心: {protein_name}")
                    except Exception as e:
                        pocket_com = pocket_coords.mean(axis=0)
                        print(f"警告: 读取配体文件 {ligand_file} 时出错: {e}，使用口袋质心: {protein_name}")
                else:
                    # 如果没有找到对应的配体文件，使用口袋质心
                    pocket_com = pocket_coords.mean(axis=0)
                    if ligand_file is None:
                        print(f"警告: 未找到 {protein_prefix} 对应的配体文件，使用口袋质心: {protein_name}")
                    else:
                        print(f"警告: 配体文件 {ligand_file} 不存在，使用口袋质心: {protein_name}")
            else:
                # 使用口袋质心
                pocket_com = pocket_coords.mean(axis=0)
            
            protein_com_map[protein_name] = pocket_com
            
            # 将坐标转换为tensor并中心化以输入模型
            pocket_pos = torch.from_numpy(pocket_coords).float()
            pocket_pos = pocket_pos - torch.from_numpy(pocket_com).float() # 使用计算出的COM进行中心化
            
            n_atoms = len(pocket_atom_symbols)
            atom_type = torch.zeros(n_atoms, len(atom_map))
            for i, symbol in enumerate(pocket_atom_symbols):
                if symbol in atom_map:
                    idx = atom_map.index(symbol)
                    atom_type[i, idx] = 1
            
            g = dgl.graph(([], []), num_nodes=n_atoms)
            g.ndata['x_1_true'] = pocket_pos
            g.ndata['a_1_true'] = atom_type
            
            if g:
                protein_graph_map[protein_name] = g # 直接存储图对象，而不是列表
                
        except Exception as e:
            print(f"处理 {pdb_file} 时出错: {str(e)}")
            continue
    
    return protein_graph_map, protein_com_map

def parse_args():
    p = argparse.ArgumentParser(description='Testing Script')
    p.add_argument('--model_dir', type=Path, help='Path to model directory', default=None)
    p.add_argument('--checkpoint', type=Path, help='Path to checkpoint file', default=None)
    p.add_argument('--output_dir', type=Path, help='Path to output directory', default=None)
    p.add_argument('--pdb', type=Path, help='Path to a single PDB file', default=None)
    p.add_argument('--pdb_dir', type=Path, help='Path to a directory containing PDB files', default=None)
    p.add_argument('--n_mols_per_protein', type=int, default=10, help='The number of molecules to generate per protein.')
    p.add_argument('--ref_ligands_dir', type=Path, help='Path to reference ligands directory (SDF files)', default=None)
    p.add_argument('--frag_file', type=Path, help='Path to fragment file', default=None)

    p.add_argument('--n_atoms_per_mol', type=int, default=None, help="The number of atoms in every molecule. If None, it will be sampled from the training data distribution.")
    p.add_argument('--n_timesteps', type=int, default=20, help="Number of timesteps for integration.")
    p.add_argument('--xt_traj', action='store_true', help='Save the x-t trajectory of the sampled molecules.')
    p.add_argument('--ep_traj', action='store_true', help='Save the endpoint trajectory of the sampled molecules.')
    p.add_argument('--metrics', action='store_true', help='Compute metrics on the sampled molecules.')
    p.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size for sampling molecules.')
    p.add_argument('--baseline_comparison', action='store_true', help='If true, output format will be a pickle file for baseline comparison.')
    
    p.add_argument('--stochasticity', type=float, default=None, help='Stochasticity for sampling molecules (for CTMC models).')
    p.add_argument('--hc_thresh', type=float, default=None, help='High confidence threshold for purity sampling (for CTMC models).')
    
    p.add_argument('--seed', type=int, default=None)

    args = p.parse_args()

    if args.model_dir is not None and args.checkpoint is not None:
        raise ValueError('Only specify model_dir or checkpoint, not both.')
    
    if args.model_dir is None and args.checkpoint is None:
        raise ValueError('Must specify model_dir or checkpoint.')

    if args.pdb is not None and args.pdb_dir is not None:
        raise ValueError('Only specify --pdb or --pdb_dir, not both.')
    
    if args.pdb is None and args.pdb_dir is None:
        raise ValueError('Must specify --pdb or --pdb_dir.')

    if args.hc_thresh is not None and (args.hc_thresh < 0 or args.hc_thresh > 1):
        raise ValueError('hc_thresh must be on the interval [0, 1].')

    return args

def shift_rdkit_mol(rdkit_mol, offset):
    """一个辅助函数，用于平移RDKit分子对象中的原子坐标。"""
    if rdkit_mol is None:
        return
    conf = rdkit_mol.GetConformer()
    for i in range(rdkit_mol.GetNumAtoms()):
        old_pos = conf.GetAtomPosition(i)
        new_pos_np = np.array([old_pos.x, old_pos.y, old_pos.z]) + offset
        new_pos_rdkit = Point3D(float(new_pos_np[0]), float(new_pos_np[1]), float(new_pos_np[2]))
        conf.SetAtomPosition(i, new_pos_rdkit)

if __name__ == "__main__":

    args = parse_args()

    visualize = args.xt_traj or args.ep_traj

    if args.seed is not None:
        seed_everything(args.seed)

    if args.model_dir is not None:
        model_dir = args.model_dir
        checkpoint_file = args.model_dir / 'checkpoints' / 'last.ckpt'
    else:
        model_dir = args.checkpoint.parent.parent
        checkpoint_file = args.checkpoint

    model = PocketFlowMol.load_from_checkpoint(checkpoint_file)
    config_file = model_dir / 'config.yaml'
    config = read_config_file(config_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    output_dir = args.output_dir

    pdb_input = args.pdb if args.pdb else args.pdb_dir
    atom_map = config['dataset']['atom_map']
    
    print("正在处理蛋白质输入并计算口袋质心...")
    protein_graph_templates, protein_com_map = process_protein_input(
        pdb_input=pdb_input,
        atom_map=atom_map,
        ref_ligands_dir=args.ref_ligands_dir
    )

    if not protein_graph_templates:
        raise ValueError("没有从输入的PDB文件/目录中生成任何有效的蛋白质图。")

    # --- 3. 设置生成任务和跟踪器 ---
    protein_names = list(protein_graph_templates.keys())
    protein_to_valid_mols = defaultdict(list)
    written_files = set() # 用于跟踪已创建的文件，以处理追加模式
    
    total_target_mols = len(protein_names) * args.n_mols_per_protein
    total_generated_count = 0
    max_total_attempts = total_target_mols * 2 # 安全限制，防止无限循环
    
    start_time = time.time()
    print(f"开始为 {len(protein_names)} 个蛋白质生成分子，每个目标 {args.n_mols_per_protein} 个。")

    # --- 4. 主要生成循环 (带重试机制) ---
    while sum(len(mols) for mols in protein_to_valid_mols.values()) < total_target_mols:
        
        if total_generated_count > max_total_attempts:
            print("\n警告：已达到最大生成尝试次数。可能某些蛋白质难以生成有效分子。")
            break

        # -- a. 动态构建批次 --
        batch_g_list = []
        batch_protein_map = [] # 记录当前批次中每个图对应的蛋白质名称
        
        # 收集所有还需要生成分子的蛋白质及其剩余需求
        remaining_proteins = []
        for protein_name in protein_names:
            remaining = args.n_mols_per_protein - len(protein_to_valid_mols[protein_name])
            if remaining > 0:
                remaining_proteins.append((protein_name, remaining))

        # 按剩余需求从大到小排序（优先处理需求大的）
        remaining_proteins.sort(key=lambda x: x[1], reverse=True)

        # 填充批次直到达到max_batch_size
        for protein_name, remaining in remaining_proteins:
            # 计算这个蛋白本次可以生成的数量
            available_slots = args.max_batch_size - len(batch_g_list)
            if available_slots <= 0:
                break
                
            # 本次为该蛋白生成的数量 = min(剩余需求, 可用槽位)
            count_this_batch = min(remaining, available_slots)
            
            # 添加相应数量的图副本到批次中
            batch_g_list.extend([protein_graph_templates[protein_name]] * count_this_batch)
            batch_protein_map.extend([protein_name] * count_this_batch)
        
        if not batch_g_list:
            print("所有蛋白质均已满足目标数量。")
            break # 所有任务完成

        # -- b. 模型采样 --
        batch_size = len(batch_g_list)
        total_generated_count += batch_size
        
        # 打印当前进度
        progress = sum(len(mols) for mols in protein_to_valid_mols.values())
        print(f"\r进度: {progress}/{total_target_mols} | "
              f"当前批次: {batch_size} 个分子 | "
              f"总尝试次数: {total_generated_count}", end="")

        pocket_g = dgl.batch(batch_g_list).to(device)
        pocket_g = add_knn_edges(pocket_g)

        if args.n_atoms_per_mol is None:
            batch_molecules: List[SampledMolecule] = model.sample_random_sizes(
                batch_size, pocket_g, device=device, n_timesteps=args.n_timesteps, 
                xt_traj=args.xt_traj, ep_traj=args.ep_traj, 
                stochasticity=args.stochasticity, high_confidence_threshold=args.hc_thresh
            )
        else:
            base = args.n_atoms_per_mol
            n_atoms = torch.randint(
                low=max(1, base - 3),  # 至少1个原子
                high=base + 7,         # randint右开区间
                size=(batch_size,),
                dtype=torch.long,
                device=device
            )
            
            if args.frag_file is not None:
                batch_molecules: List[SampledMolecule] = model.sample_with_frag(
                    n_atoms, 
                    pocket_g,
                    device=device, 
                    n_timesteps=args.n_timesteps, 
                    xt_traj=args.xt_traj,
                    ep_traj=args.ep_traj,
                    stochasticity=args.stochasticity,
                    high_confidence_threshold=args.hc_thresh,
                    frag_file=args.frag_file,
                )
            else:
                batch_molecules: List[SampledMolecule] = model.sample(
                    n_atoms, pocket_g, device=device, n_timesteps=args.n_timesteps, 
                    xt_traj=args.xt_traj, ep_traj=args.ep_traj, 
                    stochasticity=args.stochasticity, high_confidence_threshold=args.hc_thresh
                )

        # -- c. 验证、处理和分批写入 --
        newly_validated_mols_by_protein = defaultdict(list)

        # 批量验证整个批次
        batch_validation_result = validate_molecules(batch_molecules)

        # 创建有效分子的映射
        valid_mols_set = set(batch_validation_result['valid'])

        # 处理批次中的每个分子
        for i, mol in enumerate(batch_molecules):
            protein_name = batch_protein_map[i]
            
            # 如果该蛋白质已满，则跳过
            if len(protein_to_valid_mols[protein_name]) >= args.n_mols_per_protein:
                continue
            
            # 检查分子是否有效
            if mol in valid_mols_set:
                # 平移坐标
                com_offset = protein_com_map.get(protein_name)
                if com_offset is not None:
                    shift_rdkit_mol(mol.rdkit_mol, com_offset)
                    if args.xt_traj:
                        for traj_mol in mol.traj_mols: 
                            shift_rdkit_mol(traj_mol, com_offset)
                    if args.ep_traj:
                        for traj_mol in mol.ep_traj_mols: 
                            shift_rdkit_mol(traj_mol, com_offset)
                
                # 添加到主列表和待写入列表
                protein_to_valid_mols[protein_name].append(mol)
                newly_validated_mols_by_protein[protein_name].append(mol)

        # -- d. 将这个批次中新验证的分子写入文件 --
        if newly_validated_mols_by_protein:
            for protein_name, mols_to_write in newly_validated_mols_by_protein.items():
                
                # 在这里处理不同的输出格式
                if args.baseline_comparison:
                    # 对于baseline，通常是最后一起写入，所以这里可以先收集
                    # 或者，如果格式允许，也可以追加
                    pass # 暂时跳过，因为其格式要求是(mols, time)
                
                # 写入SDF文件
                output_sdf_file = output_dir / f'{protein_name}.sdf'
                write_mode = 'a' if output_sdf_file in written_files else 'w'
                
                try:
                    with open(output_sdf_file, write_mode) as f:
                        with Chem.SDWriter(f) as sdf_writer:
                            sdf_writer.SetKekulize(False)
                            for mol in mols_to_write:
                                if mol.rdkit_mol:
                                    sdf_writer.write(mol.rdkit_mol)
                    written_files.add(output_sdf_file)
                except Exception as e:
                    print(f"\n警告：写入文件 {output_sdf_file} 时出错: {e}")

                # 写入轨迹文件
                if visualize:
                    for mol in mols_to_write:
                        mol_idx = len(protein_to_valid_mols[protein_name]) - len(mols_to_write) + mols_to_write.index(mol)
                        if args.xt_traj:
                            mol_output_file = output_dir / f'{protein_name}_{mol_idx}_xt.sdf'
                            with Chem.SDWriter(str(mol_output_file)) as sdf_writer:
                                for traj_mol in mol.traj_mols: sdf_writer.write(traj_mol)
                        if args.ep_traj:
                            mol_output_file = output_dir / f'{protein_name}_{mol_idx}_ep.sdf'
                            with Chem.SDWriter(str(mol_output_file)) as sdf_writer:
                                for traj_mol in mol.ep_traj_mols: sdf_writer.write(traj_mol)

    # --- 5. 最终总结 ---
    end_time = time.time()
    sampling_time = end_time - start_time
    print(f"\n\n分子生成完成，总耗时: {sampling_time:.2f} 秒。")

    valid_count_total = 0

    for protein_name in protein_names:
        valid_count = len(protein_to_valid_mols[protein_name])
        valid_count_total = valid_count_total + valid_count
        print(f"  - 蛋白质 '{protein_name}': 生成了 {valid_count}/{args.n_mols_per_protein} 个有效分子。")


    if args.baseline_comparison:
        print("\n正在为 baseline comparison 模式写入 pkl 文件...")
        for protein_name, molecules in protein_to_valid_mols.items():
            output_file = output_dir / f'{protein_name}_baseline_comparison.pkl'
            rdkit_mols = [m.rdkit_mol for m in molecules if m.rdkit_mol is not None]
            with open(output_file, 'wb') as f:
                pickle.dump((rdkit_mols, sampling_time), f)
            print(f"  - 已写入: {output_file}")

    # if args.metrics:
    #     processed_data_dir = config['dataset']['processed_data_dir']
    #     sample_analyzer = SampleAnalyzer(processed_data_dir=Path(processed_data_dir))
    #     metrics = sample_analyzer.analyze(molecules)
    #     js_div = sample_analyzer.compute_energy_divergence(molecules)
    #     metrics['energy_js_div'] = js_div

    #     metrics_txt_file = output_dir / f'{protein_name}_metrics.txt'
    #     metrics_pkl_file = output_dir / f'{protein_name}_metrics.pkl'

    #     print(f'正在将指标写入 {metrics_txt_file} 和 {metrics_pkl_file}')
    #     with open(metrics_txt_file, 'w') as f:
    #         for k, v in metrics.items():
    #             f.write(f'{k}: {v}\n')
    #     with open(metrics_pkl_file, 'wb') as f:
    #         pickle.dump(metrics, f)

    print(f"\n所有处理完成。输出文件已保存到目录: {output_dir}, 总尝试次数：{total_generated_count}, 总生成分子{valid_count_total}, 生成成功率{valid_count_total/total_generated_count}。")