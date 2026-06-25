import sys
import os
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

def has_fragments(smiles):
    """检查SMILES是否含有断点（未连接的原子或键）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # 无效SMILES
    
    # 检查是否有未连接的原子（如 "." 分隔的片段）
    if "." in smiles:
        return True
    
    # 检查分子中是否有未连接的原子（如游离原子）
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0 and atom.GetAtomicNum() != 0:
            return True
    
    return False

def calculate_diversity_tanimoto(molecules):
    """
    计算分子集合的Tanimoto多样性指数
    使用基于Morgan指纹的Tanimoto距离
    """
    if len(molecules) <= 1:
        return 0.0
    
    # 生成Morgan指纹
    fps = []
    for mol in molecules:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(fp)
    
    if len(fps) <= 1:
        return 0.0
    
    # 计算所有分子对之间的平均Tanimoto距离
    similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(1 - sim)  # 转换为距离
    
    if similarities:
        diversity = np.mean(similarities)
        return round(diversity, 4)  # 保留4位小数
    else:
        return 0.0

def process_sdf_file(input_sdf, output_csv=None):
    """
    处理单个SDF文件，返回包含有效SMILES的DataFrame
    
    参数:
        input_sdf (str): 输入的SDF文件路径
        output_csv (str): 可选，输出的CSV文件路径
        
    返回:
        pd.DataFrame: 包含有效SMILES的DataFrame
    """

    supplier = Chem.SDMolSupplier(input_sdf)
    
    data = []
    valid_count = 0
    invalid_count = 0
    total_count = 0
    all_molecules = []
    all_smiles = []
    
    for mol in supplier:
        total_count += 1
        if mol is None:
            invalid_count += 1
            continue
        
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            if not has_fragments(smiles):
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
                
                data.append({
                    'Name': name, 
                    'smiles': smiles, 
                    'Source': os.path.basename(input_sdf)
                })
                
                valid_count += 1
                all_molecules.append(mol)
                all_smiles.append(smiles)
            else:
                invalid_count += 1
            
        except:
            invalid_count += 1
    
    df = pd.DataFrame(data)
    
    # 计算统计信息
    if total_count > 0:
        # 计算有效率
        validity_rate = round(valid_count / total_count * 100, 2)
        
        # 计算Uniqueness率
        unique_count = len(set(all_smiles)) if all_smiles else 0
        uniqueness_rate = round(unique_count / valid_count * 100, 2) if valid_count > 0 else 0
        
        # 计算Tanimoto多样性指数
        diversity_tanimoto = calculate_diversity_tanimoto(all_molecules) if all_molecules else 0
        
        print(f"\n文件 {os.path.basename(input_sdf)} 统计信息:")
        print("-" * 40)
        print(f"总分子数: {total_count}")
        print(f"有效分子数: {valid_count}")
        print(f"无效分子数: {invalid_count}")
        print(f"有效率: {validity_rate}%")
        print(f"Unique分子数: {unique_count}")
        print(f"Uniqueness率: {uniqueness_rate}%")
        print(f"Tanimoto多样性指数: {diversity_tanimoto}")
    
    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"结果已保存到: {output_csv}")
    
    return df, all_molecules, all_smiles

def sdf_to_smiles_csv(input_path, output_csv):
    """
    将SDF文件或目录下的所有SDF文件转换为CSV，仅保留无断点的SMILES
    
    参数:
        input_path (str): 输入的SDF文件路径或目录路径
        output_csv (str): 输出的CSV文件路径
    """
    try:
        all_data = []
        all_molecules = []
        all_smiles = []
        total_valid = 0
        total_invalid = 0
        total_molecules = 0
        
        if os.path.isdir(input_path):
            # 处理目录下的所有SDF文件
            print(f"正在处理目录: {input_path}")
            sdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.sdf')]
            
            if not sdf_files:
                print("目录中没有找到SDF文件！")
                return
            
            for filename in sdf_files:
                filepath = os.path.join(input_path, filename)
                df, molecules, smiles = process_sdf_file(filepath)
                
                if not df.empty:
                    all_data.append(df)
                    all_molecules.extend(molecules)
                    all_smiles.extend(smiles)
                    total_valid += len(df)
                
                # 统计总分子数
                supplier = Chem.SDMolSupplier(filepath)
                total_molecules += len(supplier)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df.to_csv(output_csv, index=False)
                
                # 计算全局统计信息
                total_invalid = total_molecules - total_valid
                global_validity_rate = round(total_valid / total_molecules * 100, 2) if total_molecules > 0 else 0
                
                global_unique_count = len(set(all_smiles)) if all_smiles else 0
                global_uniqueness_rate = round(global_unique_count / total_valid * 100, 2) if total_valid > 0 else 0
                
                global_diversity_tanimoto = calculate_diversity_tanimoto(all_molecules) if all_molecules else 0
                
                print("\n" + "="*60)
                print("全局统计信息:")
                print("="*60)
                print(f"总分子数: {total_molecules}")
                print(f"总有效分子数: {total_valid}")
                print(f"总无效分子数: {total_invalid}")
                print(f"全局有效率: {global_validity_rate}%")
                print(f"全局Unique分子数: {global_unique_count}")
                print(f"全局Uniqueness率: {global_uniqueness_rate}%")
                print(f"全局Tanimoto多样性指数: {global_diversity_tanimoto}")
                print(f"\n结果已合并保存到: {output_csv}")
                print(f"共处理 {len(sdf_files)} 个SDF文件")
            else:
                print("所有SDF文件中均未找到有效分子！")
                
        elif os.path.isfile(input_path) and input_path.lower().endswith('.sdf'):
            # 处理单个SDF文件
            df, molecules, smiles = process_sdf_file(input_path, output_csv)
            if not df.empty:
                df.to_csv(output_csv, index=False)
        else:
            print("错误: 输入路径必须是SDF文件或包含SDF文件的目录！")
            sys.exit(1)

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python sdf_to_smiles_nofrag.py 输入文件或目录 输出文件.csv")
        print("示例: python sdf_to_smiles_nofrag.py input.sdf output.csv")
        print("示例: python sdf_to_smiles_nofrag.py ./sdf_files/ output.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2]
    
    # 确保输出文件扩展名为.csv
    if not output_file.lower().endswith('.csv'):
        output_file += '.csv'
    
    sdf_to_smiles_csv(input_path, output_file)