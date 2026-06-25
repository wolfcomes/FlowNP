import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import argparse
import sys
from glob import glob

class MolecularDiversityAnalyzer:
    def __init__(self, fingerprint_type='ECFP4', radius=2, n_bits=2048):
        """
        初始化分子多样性分析器
        
        参数:
            fingerprint_type: 指纹类型 ('ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'Avalon')
            radius: ECFP指纹的半径
            n_bits: 指纹位数
        """
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        
    def read_smiles_from_csv(self, file_paths, smiles_column='smiles'):
        """
        从一个或多个CSV文件中读取SMILES数据
        
        参数:
            file_paths: CSV文件路径列表或单个文件路径
            smiles_column: SMILES列名（默认为'smiles'）
            
        返回:
            dict: 包含每个文件名的分子列表的字典
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        all_molecules = {}
        
        for file_path in file_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"错误: 文件 {file_path} 不存在")
                    continue
                    
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查SMILES列是否存在
                if smiles_column not in df.columns:
                    # 尝试自动检测SMILES列
                    possible_columns = [col for col in df.columns if 'smile' in col.lower()]
                    if possible_columns:
                        smiles_column_actual = possible_columns[0]
                        print(f"警告: 在文件 {file_path} 中未找到 '{smiles_column}' 列，使用 '{smiles_column_actual}' 代替")
                    else:
                        print(f"错误: 在文件 {file_path} 中未找到SMILES列。可用列: {list(df.columns)}")
                        continue
                else:
                    smiles_column_actual = smiles_column
                
                # 提取SMILES并转换为分子对象
                smiles_list = df[smiles_column_actual].dropna().astype(str).tolist()
                molecules = self._sanitize_smiles_list(smiles_list)
                
                file_name = os.path.basename(file_path)
                all_molecules[file_name] = molecules
                
                print(f"从 {file_name} 成功读取 {len(molecules)} 个有效分子 (总共 {len(smiles_list)} 个SMILES)")
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue
                
        return all_molecules
    
    def _sanitize_smiles_list(self, smiles_list):
        """将SMILES列表转换为有效的分子对象列表"""
        mols = []
        invalid_count = 0
        valid_smiles = []
        
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # 规范化SMILES以确保唯一性比较
                    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                    mols.append(mol)
                    valid_smiles.append(canonical_smi)
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        
        # 去重 - 基于规范化的SMILES
        unique_data = {}
        for smi, mol in zip(valid_smiles, mols):
            if smi not in unique_data:
                unique_data[smi] = mol
        
        unique_mols = list(unique_data.values())
        
        if invalid_count > 0:
            print(f"  过滤掉 {invalid_count} 个无效SMILES")
        if len(mols) - len(unique_mols) > 0:
            print(f"  去除 {len(mols) - len(unique_mols)} 个重复分子")
            
        return unique_mols
    
    def calculate_fingerprints(self, molecules):
        """为分子列表计算指纹"""
        fps = []
        for mol in molecules:
            try:
                if self.fingerprint_type.upper().startswith('ECFP'):
                    radius = int(self.fingerprint_type[-1]) if self.fingerprint_type[-1].isdigit() else self.radius
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=self.n_bits)
                elif self.fingerprint_type.upper().startswith('FCFP'):
                    radius = int(self.fingerprint_type[-1]) if self.fingerprint_type[-1].isdigit() else self.radius
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=True, nBits=self.n_bits)
                elif self.fingerprint_type.lower() == 'avalon':
                    from rdkit.Avalon import pyAvalonTools
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits)
                else:
                    raise ValueError(f"不支持的指纹类型: {self.fingerprint_type}")
                fps.append(fp)
            except Exception as e:
                print(f"计算分子指纹时出错: {e}")
                continue
        return fps
    
    def analyze_diversity(self, file_paths, smiles_column='smiles', reference_file=None):
        """
        分析一个或多个CSV文件中分子的多样性
        
        参数:
            file_paths: CSV文件路径列表
            smiles_column: SMILES列名
            reference_file: 参考集文件路径（通常是训练集）
        """
        # 读取所有分子
        all_molecules = self.read_smiles_from_csv(file_paths, smiles_column)
        
        if not all_molecules:
            print("没有成功读取任何分子数据")
            return None
        
        results = {}
        
        # 分析每个文件
        for file_name, molecules in all_molecules.items():
            print(f"\n{'='*50}")
            print(f"分析文件: {file_name}")
            print(f"{'='*50}")
            
            if len(molecules) < 2:
                print("分子数量太少，无法进行多样性分析")
                continue
                
            results[file_name] = self._analyze_single_set(molecules, file_name)
        
        # 如果有多个文件，比较它们之间的多样性
        if len(all_molecules) > 1:
            self._compare_multiple_sets(all_molecules)
        
        # 如果有参考集，进行外部比较
        if reference_file:
            self._analyze_with_reference(all_molecules, reference_file, smiles_column)
        
        return results
    
    def _analyze_single_set(self, molecules, set_name):
        """分析单个分子集的多样性"""
        # 计算指纹
        fps = self.calculate_fingerprints(molecules)
        
        if len(fps) < 2:
            print("有效指纹数量太少，无法进行多样性分析")
            return None
        
        # 基础统计
        total_smiles = len(molecules)
        validity = len(molecules) / total_smiles if total_smiles > 0 else 0
        uniqueness = len(molecules) / total_smiles if total_smiles > 0 else 0
        
        print(f"分子统计:")
        print(f"  - 有效分子数: {len(molecules)}")
        print(f"  - 有效性: {validity:.4f}")
        print(f"  - 唯一性: {uniqueness:.4f}")
        
        # 内部多样性
        int_div_mean, int_div_list = self._calculate_internal_diversity(fps)
        nn_mean, nn_list = self._calculate_1nn_diversity(fps)
        
        print(f"内部多样性指标:")
        print(f"  - 平均内部相似度: {int_div_mean:.4f}")
        print(f"  - 1-最近邻相似度: {nn_mean:.4f}")
        
        # Scaffold多样性
        scaffold_results = self._analyze_scaffold_diversity(molecules)
        
        # 可视化
        self._plot_diversity_distributions(int_div_list, nn_list, set_name)
        
        return {
            'molecules': molecules,
            'fingerprints': fps,
            'validity': validity,
            'uniqueness': uniqueness,
            'internal_diversity_mean': int_div_mean,
            '1nn_diversity_mean': nn_mean,
            'scaffold_diversity': scaffold_results
        }
    
    def _calculate_internal_diversity(self, fingerprints, sample_size=1000):
        """计算内部多样性"""
        if len(fingerprints) > sample_size:
            indices = np.random.choice(len(fingerprints), sample_size, replace=False)
            sample_fps = [fingerprints[i] for i in indices]
        else:
            sample_fps = fingerprints
            
        similarity_list = []
        for i in range(len(sample_fps)):
            for j in range(i+1, len(sample_fps)):
                try:
                    sim = DataStructs.TanimotoSimilarity(sample_fps[i], sample_fps[j])
                    similarity_list.append(sim)
                except:
                    continue
                    
        if not similarity_list:
            return 0.0, []
            
        return np.mean(similarity_list), similarity_list
    
    def _calculate_1nn_diversity(self, fingerprints):
        """计算1-最近邻多样性"""
        similarity_list = []
        for i in range(len(fingerprints)):
            similarities = []
            for j in range(len(fingerprints)):
                if i != j:
                    try:
                        sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                        similarities.append(sim)
                    except:
                        continue
            if similarities:
                similarity_list.append(np.max(similarities))
                
        if not similarity_list:
            return 0.0, []
            
        return np.mean(similarity_list), similarity_list
    
    def _analyze_scaffold_diversity(self, molecules):
        """分析骨架多样性"""
        scaffolds = {}
        scaffold_molecules = []
        
        for mol in molecules:
            try:
                scaffold = GetScaffoldForMol(mol)
                scaffold_smi = Chem.MolToSmiles(scaffold)
                if scaffold_smi in scaffolds:
                    scaffolds[scaffold_smi] += 1
                else:
                    scaffolds[scaffold_smi] = 1
                scaffold_molecules.append(scaffold_smi)
            except Exception as e:
                print(f"提取骨架时出错: {e}")
                continue
        
        unique_scaffolds = len(scaffolds)
        total_molecules = len(molecules)
        scaffold_ratio = unique_scaffolds / total_molecules if total_molecules > 0 else 0
        
        print(f"骨架多样性指标:")
        print(f"  - 唯一骨架数: {unique_scaffolds}")
        print(f"  - 总分子数: {total_molecules}")
        print(f"  - 骨架多样性比率: {scaffold_ratio:.4f}")
        
        # 显示最常见的骨架
        if scaffolds:
            print(f"  - 最常见的5个骨架:")
            sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: x[1], reverse=True)
            for i, (scaffold, count) in enumerate(sorted_scaffolds[:5]):
                print(f"      {i+1}. 出现次数: {count}")
        else:
            print(f"  - 无法提取任何骨架")
        
        return {
            'unique_scaffolds': unique_scaffolds,
            'scaffold_ratio': scaffold_ratio,
            'scaffold_distribution': scaffolds
        }
    
    def _compare_multiple_sets(self, all_molecules):
        """比较多个分子集之间的多样性"""
        print(f"\n{'='*50}")
        print("多个文件间多样性比较")
        print(f"{'='*50}")
        
        file_names = list(all_molecules.keys())
        n_files = len(file_names)
        
        # 创建比较矩阵
        comparison_matrix = np.zeros((n_files, n_files))
        
        for i in range(n_files):
            fps_i = self.calculate_fingerprints(all_molecules[file_names[i]])
            if not fps_i:
                continue
                
            for j in range(i, n_files):
                if i == j:
                    # 内部多样性
                    mean_sim, _ = self._calculate_internal_diversity(fps_i, sample_size=500)
                    comparison_matrix[i, j] = mean_sim
                else:
                    # 集合间多样性
                    fps_j = self.calculate_fingerprints(all_molecules[file_names[j]])
                    if not fps_j:
                        continue
                    inter_sim = self._calculate_inter_set_similarity(fps_i, fps_j)
                    comparison_matrix[i, j] = inter_sim
                    comparison_matrix[j, i] = inter_sim
        
        # 打印比较结果
        print("集合间相似度矩阵 (Tanimoto相似度):")
        header = " " * 15 + "".join([f"{name:15.15}" for name in file_names])
        print(header)
        for i, name in enumerate(file_names):
            print(f"{name:15.15}", end="")
            for j in range(n_files):
                print(f"{comparison_matrix[i, j]:15.4f}", end="")
            print()
    
    def _calculate_inter_set_similarity(self, fps1, fps2, sample_size=500):
        """计算两个分子集之间的相似度"""
        if len(fps1) > sample_size:
            indices1 = np.random.choice(len(fps1), sample_size, replace=False)
            sample_fps1 = [fps1[i] for i in indices1]
        else:
            sample_fps1 = fps1
            
        if len(fps2) > sample_size:
            indices2 = np.random.choice(len(fps2), sample_size, replace=False)
            sample_fps2 = [fps2[i] for i in indices2]
        else:
            sample_fps2 = fps2
        
        similarity_list = []
        for fp1 in sample_fps1:
            try:
                similarities = DataStructs.BulkTanimotoSimilarity(fp1, sample_fps2)
                similarity_list.append(np.max(similarities))
            except:
                continue
        
        if not similarity_list:
            return 0.0
            
        return np.mean(similarity_list)
    
    def _analyze_with_reference(self, all_molecules, reference_file, smiles_column):
        """与参考集进行比较分析"""
        print(f"\n{'='*50}")
        print("与参考集比较分析")
        print(f"{'='*50}")
        
        # 读取参考集
        reference_mols = self.read_smiles_from_csv([reference_file], smiles_column)
        if not reference_mols:
            print("无法读取参考集文件")
            return
            
        ref_name, ref_molecules = list(reference_mols.items())[0]
        ref_fps = self.calculate_fingerprints(ref_molecules)
        
        if not ref_fps:
            print("参考集无法计算指纹")
            return
        
        for file_name, molecules in all_molecules.items():
            print(f"\n分析 {file_name} 与参考集 {ref_name} 的相似度:")
            
            gen_fps = self.calculate_fingerprints(molecules)
            
            if not gen_fps:
                print("  无法计算生成集的指纹")
                continue
            
            # 计算到参考集的最近邻相似度
            nn_similarities = []
            for fp_gen in gen_fps:
                try:
                    similarities = DataStructs.BulkTanimotoSimilarity(fp_gen, ref_fps)
                    nn_similarities.append(np.max(similarities))
                except:
                    continue
            
            if not nn_similarities:
                print("  无法计算最近邻相似度")
                continue
                
            mean_nn_sim = np.mean(nn_similarities)
            std_nn_sim = np.std(nn_similarities)
            
            print(f"  - 平均最近邻相似度: {mean_nn_sim:.4f} ± {std_nn_sim:.4f}")
            print(f"  - 相似度范围: [{np.min(nn_similarities):.4f}, {np.max(nn_similarities):.4f}]")
            
            # 可视化
            self._plot_reference_comparison(nn_similarities, file_name, ref_name)
    
    def _plot_diversity_distributions(self, int_div_list, nn_list, set_name):
        """绘制多样性分布图"""
        if not int_div_list or not nn_list:
            print("  没有足够的数据进行可视化")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 内部相似度分布
        sns.histplot(int_div_list, bins=50, kde=True, ax=ax1)
        ax1.set_title(f'{set_name}\n internal similarity')
        ax1.set_xlabel('similarity')
        ax1.set_ylabel('frequency')
        
        # 1-最近邻相似度分布
        sns.histplot(nn_list, bins=50, kde=True, ax=ax2)
        ax2.set_title(f'{set_name}\n1-nn similarity')
        ax2.set_xlabel('similarity')
        ax2.set_ylabel('frequency')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_reference_comparison(self, nn_similarities, gen_name, ref_name):
        """绘制与参考集的比较图"""
        if not nn_similarities:
            return
            
        plt.figure(figsize=(8, 6))
        sns.histplot(nn_similarities, bins=50, kde=True)
        plt.title(f'{gen_name} vs {ref_name}\n1-nn similarity')
        plt.xlabel('similarity')
        plt.ylabel('frequency')
        plt.show()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='分子多样性分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 分析单个文件
  python diversity_analysis.py generated_molecules.csv
  
  # 分析多个文件
  python diversity_analysis.py file1.csv file2.csv file3.csv
  
  # 分析文件并与参考集比较
  python diversity_analysis.py generated.csv -r training_set.csv
  
  # 指定SMILES列名
  python diversity_analysis.py data.csv -c SMILES
  
  # 使用不同的指纹类型
  python diversity_analysis.py data.csv -f ECFP6
        '''
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='要分析的CSV文件路径（支持通配符）'
    )
    
    parser.add_argument(
        '-r', '--reference',
        dest='reference_file',
        help='参考集文件路径（通常是训练集）'
    )
    
    parser.add_argument(
        '-c', '--column',
        dest='smiles_column',
        default='smiles',
        help='SMILES列名（默认: smiles）'
    )
    
    parser.add_argument(
        '-f', '--fingerprint',
        dest='fingerprint_type',
        default='ECFP4',
        choices=['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'Avalon'],
        help='指纹类型（默认: ECFP4）'
    )
    
    parser.add_argument(
        '-b', '--bits',
        dest='n_bits',
        type=int,
        default=2048,
        help='指纹位数（默认: 2048）'
    )
    
    return parser.parse_args()


def expand_file_paths(file_paths):
    """扩展文件路径，支持通配符"""
    expanded_paths = []
    for path in file_paths:
        if '*' in path or '?' in path:
            # 使用通配符匹配
            matched_files = glob(path)
            if matched_files:
                expanded_paths.extend(matched_files)
                print(f"匹配到文件: {matched_files}")
            else:
                print(f"警告: 通配符路径 {path} 没有匹配到任何文件")
        else:
            # 普通文件路径
            if os.path.exists(path):
                expanded_paths.append(path)
            else:
                print(f"警告: 文件 {path} 不存在")
    
    return expanded_paths


def main():
    """主函数"""
    args = parse_arguments()
    
    # 扩展文件路径（支持通配符）
    input_files = expand_file_paths(args.files)
    
    if not input_files:
        print("错误: 没有找到有效的输入文件")
        sys.exit(1)
    
    print(f"开始分析 {len(input_files)} 个文件:")
    for file_path in input_files:
        print(f"  - {file_path}")
    
    # 检查参考集文件是否存在
    if args.reference_file and not os.path.exists(args.reference_file):
        print(f"错误: 参考集文件 {args.reference_file} 不存在")
        sys.exit(1)
    
    # 初始化分析器
    analyzer = MolecularDiversityAnalyzer(
        fingerprint_type=args.fingerprint_type,
        n_bits=args.n_bits
    )
    
    # 执行分析
    try:
        results = analyzer.analyze_diversity(
            file_paths=input_files,
            smiles_column=args.smiles_column,
            reference_file=args.reference_file
        )
        
        if results:
            print(f"\n{'='*50}")
            print("分析完成!")
            print(f"{'='*50}")
        else:
            print("分析失败，没有有效结果")
            sys.exit(1)
            
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()