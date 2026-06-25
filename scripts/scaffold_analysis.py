#!/usr/bin/env python3
"""
分子骨架多样性分析脚本
用于分析多个CSV文件中的分子骨架分布和多样性
支持命令行参数输入和输出配置
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
import os
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ScaffoldAnalyzer:
    def __init__(self):
        self.results = {}
        
    def get_murcko_scaffold(self, mol):
        """获取Murcko骨架"""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    def get_basic_scaffold(self, mol):
        """获取更简化的骨架（去除侧链原子）"""
        try:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    def read_molecules_from_csv(self, csv_files: List[str], smiles_column: str = 'smiles', 
                               name_columns: List[str] = None) -> Dict:
        """从CSV文件读取分子"""
        molecules = {}
        
        for csv_file in csv_files:
            file_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"Reading molecules from {csv_file}...")
            
            try:
                df = pd.read_csv(csv_file)
                
                # 自动检测SMILES列
                if smiles_column not in df.columns:
                    possible_columns = [col for col in df.columns if 'smile' in col.lower()]
                    if possible_columns:
                        smiles_column_actual = possible_columns[0]
                        print(f"  Using detected SMILES column: '{smiles_column_actual}'")
                    else:
                        print(f"  Warning: No SMILES column found in {csv_file}")
                        print(f"  Available columns: {list(df.columns)}")
                        continue
                else:
                    smiles_column_actual = smiles_column
                
                valid_molecules = []
                for idx, row in df.iterrows():
                    try:
                        smiles = row[smiles_column_actual]
                        if pd.isna(smiles):
                            continue
                            
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol is not None:
                            # 创建分子信息字典
                            mol_info = {
                                'mol': mol,
                                'smiles': smiles,
                                'source': file_name
                            }
                            
                            # 添加其他指定的列
                            if name_columns:
                                for col in name_columns:
                                    if col in df.columns and not pd.isna(row[col]):
                                        mol_info[col] = row[col]
                            
                            valid_molecules.append(mol_info)
                            
                    except Exception as e:
                        continue
                
                molecules[file_name] = valid_molecules
                print(f"  ✓ Loaded {len(valid_molecules)} valid molecules from {file_name}")
                
            except Exception as e:
                print(f"  ✗ Error reading {csv_file}: {e}")
        
        return molecules
    
    def analyze_scaffolds(self, molecules_dict: Dict) -> Dict:
        """分析骨架多样性"""
        analysis_results = {}
        all_scaffolds = []
        
        for source, molecules in molecules_dict.items():
            print(f"\nAnalyzing scaffolds for {source}...")
            
            if not molecules:
                print(f"  ✗ No molecules to analyze for {source}")
                continue
                
            # 提取骨架
            murcko_scaffolds = []
            basic_scaffolds = []
            scaffold_to_molecules = defaultdict(list)
            
            for mol_info in molecules:
                mol = mol_info['mol']
                murcko_scaffold = self.get_murcko_scaffold(mol)
                basic_scaffold = self.get_basic_scaffold(mol)
                
                if murcko_scaffold:
                    murcko_scaffolds.append(murcko_scaffold)
                    scaffold_to_molecules[murcko_scaffold].append(mol_info)
                
                if basic_scaffold:
                    basic_scaffolds.append(basic_scaffold)
            
            # 计算统计量
            murcko_counter = Counter(murcko_scaffolds)
            basic_counter = Counter(basic_scaffolds)
            
            # 骨架多样性指标
            total_molecules = len(molecules)
            unique_murcko = len(murcko_counter)
            unique_basic = len(basic_counter)
            
            # 计算Gini系数（衡量分布均匀性）
            gini_murcko = self.calculate_gini_coefficient(list(murcko_counter.values()))
            gini_basic = self.calculate_gini_coefficient(list(basic_counter.values()))
            
            # 最常见的骨架
            top_murcko = murcko_counter.most_common(10)
            top_basic = basic_counter.most_common(10)
            
            # 骨架覆盖率
            coverage_5 = self.calculate_coverage(murcko_counter, 0.05)  # 前5%骨架覆盖的分子比例
            coverage_10 = self.calculate_coverage(murcko_counter, 0.10)  # 前10%骨架覆盖的分子比例
            
            results = {
                'total_molecules': total_molecules,
                'unique_murcko_scaffolds': unique_murcko,
                'unique_basic_scaffolds': unique_basic,
                'murcko_diversity_ratio': unique_murcko / total_molecules if total_molecules > 0 else 0,
                'basic_diversity_ratio': unique_basic / total_molecules if total_molecules > 0 else 0,
                'gini_murcko': gini_murcko,
                'gini_basic': gini_basic,
                'top_murcko_scaffolds': top_murcko,
                'top_basic_scaffolds': top_basic,
                'scaffold_coverage_5%': coverage_5,
                'scaffold_coverage_10%': coverage_10,
                'scaffold_distribution': dict(murcko_counter),
                'scaffold_to_molecules': dict(scaffold_to_molecules)
            }
            
            analysis_results[source] = results
            all_scaffolds.extend(murcko_scaffolds)
            
            print(f"  ✓ Total molecules: {total_molecules}")
            print(f"  ✓ Unique Murcko scaffolds: {unique_murcko}")
            print(f"  ✓ Murcko diversity ratio: {results['murcko_diversity_ratio']:.3f}")
            print(f"  ✓ Gini coefficient: {gini_murcko:.3f}")
            print(f"  ✓ Top 5% scaffolds cover: {coverage_5:.1f}% of molecules")
        
        # 计算不同数据集间的骨架重叠
        if len(molecules_dict) > 1:
            print("\nCalculating scaffold overlap between datasets...")
            overlap_results = self.calculate_scaffold_overlap(analysis_results)
            analysis_results['_overlap'] = overlap_results
        
        return analysis_results
    
    def calculate_gini_coefficient(self, values):
        """计算基尼系数，衡量分布不均匀性"""
        if not values:
            return 0
        
        values = sorted(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    
    def calculate_coverage(self, counter, top_fraction):
        """计算前top_fraction比例的骨架覆盖的分子比例"""
        if not counter:
            return 0
        
        total_molecules = sum(counter.values())
        sorted_scaffolds = counter.most_common()
        
        n_top = max(1, int(len(sorted_scaffolds) * top_fraction))
        top_molecules = sum(count for _, count in sorted_scaffolds[:n_top])
        
        return (top_molecules / total_molecules) * 100
    
    def calculate_scaffold_overlap(self, analysis_results):
        """计算不同数据集间的骨架重叠"""
        sources = [s for s in analysis_results.keys() if s != '_overlap']
        overlap_matrix = np.zeros((len(sources), len(sources)))
        jaccard_matrix = np.zeros((len(sources), len(sources)))
        
        scaffold_sets = {}
        for i, source in enumerate(sources):
            scaffold_sets[source] = set(analysis_results[source]['scaffold_distribution'].keys())
        
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                set1 = scaffold_sets[source1]
                set2 = scaffold_sets[source2]
                
                if i == j:
                    overlap_matrix[i, j] = len(set1)
                    jaccard_matrix[i, j] = 1.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    
                    overlap_matrix[i, j] = intersection
                    jaccard_matrix[i, j] = intersection / union if union > 0 else 0
        
        return {
            'overlap_matrix': overlap_matrix,
            'jaccard_matrix': jaccard_matrix,
            'sources': sources
        }
    
    def generate_report(self, analysis_results, output_dir: str = "scaffold_analysis", 
                       make_plots: bool = True, save_details: bool = True):
        """生成分析报告和图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating report in {output_dir}...")
        
        # 生成汇总表格
        self._create_summary_table(analysis_results, output_dir)
        
        # 生成图表
        if make_plots:
            self._create_visualizations(analysis_results, output_dir)
        
        # 生成详细数据
        if save_details:
            self._save_detailed_data(analysis_results, output_dir)
        
        print(f"✓ Analysis complete! Results saved to {output_dir}/")
    
    def _create_summary_table(self, analysis_results, output_dir):
        """创建汇总表格"""
        summary_data = []
        
        for source, results in analysis_results.items():
            if source == '_overlap':
                continue
                
            summary_data.append({
                'Dataset': source,
                'Total Molecules': results['total_molecules'],
                'Unique Murcko Scaffolds': results['unique_murcko_scaffolds'],
                'Murcko Diversity Ratio': f"{results['murcko_diversity_ratio']:.3f}",
                'Gini Coefficient': f"{results['gini_murcko']:.3f}",
                'Coverage (Top 5%)': f"{results['scaffold_coverage_5%']:.1f}%",
                'Coverage (Top 10%)': f"{results['scaffold_coverage_10%']:.1f}%",
                'Most Common Scaffold Count': results['top_murcko_scaffolds'][0][1] if results['top_murcko_scaffolds'] else 0
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(f"{output_dir}/scaffold_summary.csv", index=False)
        print(f"✓ Summary table saved to {output_dir}/scaffold_summary.csv")
        
        # 打印到控制台
        print("\n" + "="*80)
        print("SCAFFOLD DIVERSITY SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        print("="*80)
    
    def _create_visualizations(self, analysis_results, output_dir):
        """创建可视化图表"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            sources = [s for s in analysis_results.keys() if s != '_overlap']
            
            if not sources:
                return
            
            # 1. 骨架多样性对比
            diversity_ratios = [analysis_results[s]['murcko_diversity_ratio'] for s in sources]
            
            bars = axes[0, 0].bar(sources, diversity_ratios, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Murcko Scaffold Diversity Ratio\n(Unique Scaffolds / Total Molecules)', fontsize=12)
            axes[0, 0].set_ylabel('Diversity Ratio')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 在柱状图上添加数值
            for bar, value in zip(bars, diversity_ratios):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 2. 基尼系数对比
            gini_values = [analysis_results[s]['gini_murcko'] for s in sources]
            bars = axes[0, 1].bar(sources, gini_values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Gini Coefficient\n(Lower = More Uniform Distribution)', fontsize=12)
            axes[0, 1].set_ylabel('Gini Coefficient')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, gini_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 3. 骨架分布（前15个）
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for idx, source in enumerate(sources[:4]):  # 最多显示4个数据集
                dist = analysis_results[source]['scaffold_distribution']
                top_scaffolds = Counter(dist).most_common(15)
                scaffold_names = [f"S{i+1}" for i in range(len(top_scaffolds))]
                counts = [count for _, count in top_scaffolds]
                
                x_pos = np.arange(len(scaffold_names)) + idx * 0.2
                axes[1, 0].bar(x_pos, counts, width=0.2, alpha=0.7, label=source, color=colors[idx % len(colors)])
            
            axes[1, 0].set_title('Top 15 Scaffold Distribution', fontsize=12)
            axes[1, 0].set_ylabel('Number of Molecules')
            axes[1, 0].set_xlabel('Scaffold Rank')
            axes[1, 0].legend()
            
            # 4. 数据集间重叠（热图）
            if '_overlap' in analysis_results:
                overlap_data = analysis_results['_overlap']
                jaccard_matrix = overlap_data['jaccard_matrix']
                sources = overlap_data['sources']
                
                im = axes[1, 1].imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1)
                axes[1, 1].set_xticks(range(len(sources)))
                axes[1, 1].set_yticks(range(len(sources)))
                axes[1, 1].set_xticklabels(sources, rotation=45)
                axes[1, 1].set_yticklabels(sources)
                axes[1, 1].set_title('Scaffold Similarity (Jaccard Index)', fontsize=12)
                
                # 添加颜色条
                plt.colorbar(im, ax=axes[1, 1])
                
                # 添加数值标签
                for i in range(len(sources)):
                    for j in range(len(sources)):
                        color = "white" if jaccard_matrix[i, j] > 0.5 else "black"
                        text = axes[1, 1].text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                                              ha="center", va="center", color=color, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/scaffold_analysis_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualizations saved to {output_dir}/scaffold_analysis_plots.png")
            
        except Exception as e:
            print(f"  ✗ Error creating visualizations: {e}")
    
    def _save_detailed_data(self, analysis_results, output_dir):
        """保存详细数据"""
        try:
            for source, results in analysis_results.items():
                if source == '_overlap':
                    continue
                
                # 保存骨架分布详情
                scaffold_data = []
                for scaffold, count in results['scaffold_distribution'].items():
                    scaffold_data.append({
                        'scaffold_smiles': scaffold,
                        'molecule_count': count,
                        'percentage': (count / results['total_molecules']) * 100
                    })
                
                df_scaffolds = pd.DataFrame(scaffold_data)
                df_scaffolds = df_scaffolds.sort_values('molecule_count', ascending=False)
                df_scaffolds.to_csv(f"{output_dir}/{source}_scaffold_details.csv", index=False)
            
            # 保存重叠矩阵
            if '_overlap' in analysis_results:
                overlap_data = analysis_results['_overlap']
                df_overlap = pd.DataFrame(
                    overlap_data['overlap_matrix'],
                    index=overlap_data['sources'],
                    columns=overlap_data['sources']
                )
                df_overlap.to_csv(f"{output_dir}/scaffold_overlap_matrix.csv")
                
                df_jaccard = pd.DataFrame(
                    overlap_data['jaccard_matrix'],
                    index=overlap_data['sources'],
                    columns=overlap_data['sources']
                )
                df_jaccard.to_csv(f"{output_dir}/scaffold_jaccard_similarity.csv")
            
            print(f"✓ Detailed data saved to {output_dir}/")
            
        except Exception as e:
            print(f"  ✗ Error saving detailed data: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze molecular scaffold diversity across multiple CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 基本用法 - 分析多个CSV文件
  python scaffold_analyzer.py dataset1.csv dataset2.csv dataset3.csv
  
  # 指定SMILES列名和输出目录
  python scaffold_analyzer.py data1.csv data2.csv --smiles_column canonical_smiles --output_dir my_analysis
  
  # 包含分子名称和活性信息，不生成图表
  python scaffold_analyzer.py compounds.csv --name_columns name activity --no_plots
  
  # 只生成汇总表格，不保存详细数据
  python scaffold_analyzer.py data.csv --no_details
        '''
    )
    
    # 必需参数
    parser.add_argument('csv_files', nargs='+', 
                       help='Input CSV files containing SMILES strings')
    
    # 可选参数
    parser.add_argument('--smiles_column', default='smiles',
                       help='Name of the SMILES column (default: "smiles")')
    
    parser.add_argument('--name_columns', nargs='+',
                       help='Additional columns to include in analysis (e.g., name, activity)')
    
    parser.add_argument('--output_dir', default='scaffold_analysis',
                       help='Output directory for results (default: "scaffold_analysis")')
    
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating visualization plots')
    
    parser.add_argument('--no_details', action='store_true',
                       help='Skip saving detailed scaffold data')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # 验证输入文件
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Error: Input file '{csv_file}' does not exist!")
            sys.exit(1)
    
    if args.verbose:
        print("Scaffold Diversity Analysis")
        print("=" * 50)
        print(f"Input files: {args.csv_files}")
        print(f"SMILES column: {args.smiles_column}")
        print(f"Output directory: {args.output_dir}")
        print(f"Generate plots: {not args.no_plots}")
        print(f"Save details: {not args.no_details}")
        print()
    
    # 初始化分析器
    analyzer = ScaffoldAnalyzer()
    
    # 读取分子
    molecules = analyzer.read_molecules_from_csv(
        args.csv_files, 
        args.smiles_column, 
        args.name_columns
    )
    
    if not molecules:
        print("Error: No valid molecules found in the provided files!")
        print("Please check:")
        print("  1. File paths are correct")
        print("  2. SMILES column exists (use --smiles_column if different from 'smiles')")
        print("  3. Files contain valid SMILES strings")
        sys.exit(1)
    
    # 分析骨架
    results = analyzer.analyze_scaffolds(molecules)
    
    # 生成报告
    analyzer.generate_report(
        results, 
        args.output_dir,
        make_plots=not args.no_plots,
        save_details=not args.no_details
    )

if __name__ == "__main__":
    main()