import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List
import warnings
from Bio.PDB import PDBParser
import numpy as np
import torch
import tqdm
import yaml
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool
import pandas as pd
import random
# 抑制RDKit警告
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)

from FlowNP.src.data_processing.geom import MoleculeFeaturizer
from src.utils.dataset_stats import compute_p_c_given_a

def chunks(lst, n):
    """将列表 lst 分割成大小为 n 的子列表。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description='Process CrossDocked protein-ligand pairs')
    p.add_argument('--config', type=Path, required=True, help='配置文件路径')
    p.add_argument('--dist_cutoff', type=float, default=15.0, help='定义口袋的距离阈值 (埃)')
    p.add_argument('--chunk_size', type=int, default=1000, help='一次处理的分子数量')
    p.add_argument('--n_cpus', type=int, default=1, help='计算部分电荷时使用的CPU核心数')
    p.add_argument('--max_pairs', type=int, default=None, help='要处理的最大蛋白-配体对数（用于调试）')
    p.add_argument('--max_ligand_atoms', type=int, default=100, help='配体分子的最大原子数')
    p.add_argument('--explicit_aromaticity', action='store_true', help='使用显式芳香性而不是Kekulization')
    
    args = p.parse_args()
    return args

def extract_pocket_atoms(pdb_file, ligand_coords, cutoff, atom_map):
    """从PDB文件中提取距离配体给定阈值内的口袋原子。"""
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    pocket_coords = []
    pocket_atom_symbols = []

    # Standard amino acid residue names
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    }

    for residue in structure.get_residues():
        resname = residue.get_resname().strip()
        
        # Skip water molecules and non-standard residues
        if resname == 'HOH' or resname not in standard_residues:
            continue

        res_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
        
        # Calculate minimum distance between residue atoms and ligand atoms
        min_dist = np.min(np.sqrt(np.sum((res_coords[:, None, :] - ligand_coords[None, :, :])**2, axis=-1)))
        
        if min_dist < cutoff:
            for atom in residue.get_atoms():
                atom_symbol = atom.element.strip().capitalize()
                if atom_symbol in atom_map and atom_symbol != 'H':
                    pocket_coords.append(atom.get_coord())
                    pocket_atom_symbols.append(atom_symbol)
    
    return np.array(pocket_coords), pocket_atom_symbols


def load_crossdocked_data(config, dist_cutoff, max_pairs_per_split=None):
    """从CrossDocked数据集中加载蛋白-配体对和数据划分。"""
    basedir = Path(config['dataset']['raw_data_dir'])
    datadir = basedir / 'crossdocked_pocket10'
    split_path = basedir / 'split_by_name.pt'
    atom_map = config['dataset']['atom_map']
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found at {split_path}")
    if not datadir.is_dir():
        raise FileNotFoundError(f"CrossDocked data directory not found at {datadir}")

    print(f"从 {split_path} 加载数据划分")
    data_splits = torch.load(split_path)
    
    if 'val' not in data_splits:
        print("从训练集中创建验证集...")
        random.seed(42)
        # 验证集样本数量可以根据需要调整
        val_sample_size = min(10000, len(data_splits['train']) // 10)
        data_splits['val'] = random.sample(data_splits['train'], val_sample_size)
        # 从训练集中移除验证集样本
        data_splits['train'] = [item for item in data_splits['train'] if item not in data_splits['val']]

    all_data = {}
    for split_name, split_pairs in data_splits.items():
        print(f"为 {split_name} 划分加载蛋白-配体对...")
        pair_data_list = []
        
        if max_pairs_per_split and len(split_pairs) > max_pairs_per_split:
            split_pairs = split_pairs[:max_pairs_per_split]

        pbar = tqdm.tqdm(split_pairs, desc=f"加载 {split_name} 对")
        for pocket_fn, ligand_fn in pbar:
            sdffile = datadir / ligand_fn
            pdbfile = datadir / pocket_fn
            
            # 1. 加载配体
            mol = Chem.SDMolSupplier(str(sdffile), removeHs=False, sanitize=False)[0]
            if mol is None: continue
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            
            # 2. 获取配体坐标
            ligand_coords = mol.GetConformer().GetPositions()

            # 3. 提取口袋原子
            pocket_coords, pocket_atom_symbols = extract_pocket_atoms(pdbfile, ligand_coords, dist_cutoff, atom_map)

            # 如果没有找到口袋原子，则跳过该对
            if len(pocket_coords) == 0:
                continue

            pair_data_list.append({
                'ligand_mol': mol,
                'pocket_coords': pocket_coords,
                'pocket_atom_symbols': pocket_atom_symbols,
                'pair_id': f"{pocket_fn}_{ligand_fn}"
            })


        all_data[split_name] = pair_data_list
        print(f"为 {split_name} 划分加载了 {len(pair_data_list)} 个有效的蛋白-配体对。")
        
    return all_data

def process_split(pair_data_list, split_name, args, config):
    """处理一个数据划分（train/val/test）并保存处理后的数据。"""
    
    dataset_config = config['dataset']
    atom_map = dataset_config['atom_map']
    output_dir = Path(dataset_config['processed_data_dir'])
    output_dir.mkdir(exist_ok=True, parents=True) 

    if not pair_data_list:
        print(f"跳过 {split_name}，因为它不包含有效的数据对。")
        return

    print(f"正在处理 {split_name} 划分，包含 {len(pair_data_list)} 个蛋白-配体对")

    # 分离数据以便批处理
    ligand_molecules = [p['ligand_mol'] for p in pair_data_list]
    smiles_list = [Chem.MolToSmiles(m) for m in ligand_molecules]
    pair_ids = [p['pair_id'] for p in pair_data_list]

    # 初始化用于存储配体特征的列表
    all_ligand_positions, all_ligand_atom_types, all_ligand_atom_charges = [], [], []
    all_ligand_bond_types, all_ligand_bond_idxs = [], []
    all_ligand_bond_order_counts = torch.zeros(5 if args.explicit_aromaticity else 4, dtype=torch.int64)

    # 初始化用于存储口袋特征的列表
    all_pocket_positions = [torch.from_numpy(p['pocket_coords']).float() for p in pair_data_list]
    all_pocket_atom_types = []

    # 为口袋原子创建One-hot编码
    for pair in pair_data_list:
        n_atoms = len(pair['pocket_atom_symbols'])
        atom_type = torch.zeros(n_atoms, len(atom_map))
        for i, symbol in enumerate(pair['pocket_atom_symbols']):
            if symbol in atom_map:
                idx = atom_map.index(symbol)  # Find position of symbol in the list
                atom_type[i, idx] = 1
        all_pocket_atom_types.append(atom_type)

    # 使用MoleculeFeaturizer处理配体
    mol_featurizer = MoleculeFeaturizer(
        atom_map, 
        n_cpus=args.n_cpus, 
        max_atoms=args.max_ligand_atoms,
        explicit_aromaticity=args.explicit_aromaticity
    )

    # ====== 修改开始：修复enumerate对象处理 ======
    # 创建索引列表
    indices = list(range(len(ligand_molecules)))
    
    # 分块处理索引和分子
    chunk_size = args.chunk_size
    n_chunks = (len(ligand_molecules) - 1) // chunk_size + 1
    tqdm_iterator = tqdm.tqdm(range(n_chunks), desc=f'特征化 {split_name} 配体')
    
    failed_molecules = 0
    success_indices = []
    
    for chunk_idx in tqdm_iterator:
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(ligand_molecules))
        
        # 获取当前批次的索引和分子
        batch_indices = indices[start:end]
        batch_mols = [ligand_molecules[i] for i in batch_indices]
        
        # 特征化当前批次的分子
        pos, types, charges, b_types, b_idxs, failed_idx, b_counts = mol_featurizer.featurize_molecules(batch_mols)
        failed_molecules += len(failed_idx)
        success_indices_in_chunk = [i for i in range(len(batch_mols)) if i not in failed_idx]
        
        # 记录成功样本的索引和特征
        for i, (idx, p, t, c, bt, bidx) in enumerate(zip(success_indices_in_chunk, pos, types, charges, b_types, b_idxs)):
            if p is not None:  # 检查是否成功处理
                all_ligand_positions.append(p)
                all_ligand_atom_types.append(t)
                all_ligand_atom_charges.append(c)
                all_ligand_bond_types.append(bt)
                all_ligand_bond_idxs.append(bidx)
                success_indices.append(idx)
        
        all_ligand_bond_order_counts += b_counts
    # ====== 修改结束 ======

    # 根据成功索引过滤口袋数据
    all_pocket_positions = [all_pocket_positions[i] for i in success_indices]
    all_pocket_atom_types = [all_pocket_atom_types[i] for i in success_indices]
    pair_ids = [pair_ids[i] for i in success_indices]
    smiles_list = [smiles_list[i] for i in success_indices]
    # ====== 修改结束 ======

    # --- 数据拼接和索引创建 ---
    # ...（后续代码保持不变，使用过滤后的数据）...
    # 注意：以下代码使用过滤后的新列表，确保口袋和配体数量一致
    
    # 配体
    n_ligand_atoms_list = torch.tensor([x.shape[0] for x in all_ligand_positions])
    n_ligand_bonds_list = torch.tensor([x.shape[0] for x in all_ligand_bond_idxs])
    ligand_node_idx = torch.zeros((len(n_ligand_atoms_list), 2), dtype=torch.int64)
    ligand_node_idx[:, 1] = torch.cumsum(n_ligand_atoms_list, dim=0)
    ligand_node_idx[1:, 0] = ligand_node_idx[:-1, 1]
    ligand_edge_idx = torch.zeros((len(n_ligand_bonds_list), 2), dtype=torch.int64)
    ligand_edge_idx[:, 1] = torch.cumsum(n_ligand_bonds_list, dim=0)
    ligand_edge_idx[1:, 0] = ligand_edge_idx[:-1, 1]

    # 口袋
    n_pocket_atoms_list = torch.tensor([x.shape[0] for x in all_pocket_positions])
    pocket_node_idx = torch.zeros((len(n_pocket_atoms_list), 2), dtype=torch.int64)
    pocket_node_idx[:, 1] = torch.cumsum(n_pocket_atoms_list, dim=0)
    pocket_node_idx[1:, 0] = pocket_node_idx[:-1, 1]

    # 拼接所有张量
    data_dict = {
        'pair_ids': pair_ids,
        'smiles': smiles_list,
        'ligand_positions': torch.cat(all_ligand_positions, dim=0).float(),
        'ligand_atom_types': torch.cat(all_ligand_atom_types, dim=0),
        'ligand_atom_charges': torch.cat(all_ligand_atom_charges, dim=0).int(),
        'ligand_bond_types': torch.cat(all_ligand_bond_types, dim=0),
        'ligand_bond_idxs': torch.cat(all_ligand_bond_idxs, dim=0).long(),
        'ligand_node_idx_array': ligand_node_idx,
        'ligand_edge_idx_array': ligand_edge_idx,
        'pocket_positions': torch.cat(all_pocket_positions, dim=0).float(),
        'pocket_atom_types': torch.cat(all_pocket_atom_types, dim=0),
        'pocket_node_idx_array': pocket_node_idx
    }

    # 保存主数据字典
    output_file = output_dir / f'{split_name}_processed.pt'
    torch.save(data_dict, output_file)

    # create histogram of number of atoms
    n_ligand_atoms, counts = torch.unique(n_ligand_atoms_list, return_counts=True)
    histogram_file = output_dir / f'{split_name}_n_atoms_histogram.pt'
    torch.save((n_ligand_atoms, counts), histogram_file)

    joint_distribution = torch.stack([n_ligand_atoms_list, n_pocket_atoms_list], dim=1)

    # # 计算每种(配体原子数, 口袋原子数)组合的出现次数
    unique_pairs, pair_counts = torch.unique(joint_distribution, dim=0, return_counts=True)

    # 保存联合分布
    joint_dist_file = output_dir / f'{split_name}_joint_atoms_distribution.pt'
    torch.save({'unique_pairs': unique_pairs, 'counts': pair_counts}, joint_dist_file)

    # --- 计算并保存统计数据 (仅基于配体) ---
    p_a = data_dict['ligand_atom_types'].sum(dim=0).float()
    p_a = p_a / p_a.sum()
    p_e = all_ligand_bond_order_counts.float() / all_ligand_bond_order_counts.sum()
    charge_vals, charge_counts = torch.unique(data_dict['ligand_atom_charges'], return_counts=True)
    p_c = torch.zeros(5, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        if -2 <= c_val <= 2: p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()
    p_c_given_a = compute_p_c_given_a(data_dict['ligand_atom_charges'], data_dict['ligand_atom_types'], atom_map)

    marginal_dists_file = output_dir / f'{split_name}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)

    smiles_file = output_dir / f'{split_name}_smiles.pkl'
    with open(smiles_file, 'wb') as f: pickle.dump(smiles_list, f)

    print(f"完成处理 {split_name} 划分: {len(pair_ids)} 个数据对, {failed_molecules} 个失败的配体")

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['dataset']['dataset_name'] != 'crossdock':
        raise ValueError('此脚本配置为处理CrossDocked数据集，请检查配置文件中的dataset_name')

    # 加载所有数据
    all_data = load_crossdocked_data(config, args.dist_cutoff, max_pairs_per_split=args.max_pairs)

    # 处理每个数据划分
    for split_name, pair_data_list in all_data.items():
        output_name_map = {'train': 'train_data', 'val': 'val_data', 'test': 'test_data'}
        if split_name not in output_name_map:
            print(f"跳过无法识别的划分: {split_name}")
            continue
        
        output_split_name = output_name_map[split_name]
        
        process_split(
            pair_data_list=pair_data_list,
            split_name=output_split_name,
            args=args,
            config=config
        )

    print("\nCrossDocked数据集蛋白-配体对处理完成！")