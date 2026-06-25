import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

class MorganEnvironmentComparator:
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.bit_meanings = {}  # Store bit to substructure mapping
    
    def get_morgan_environments(self, smiles):
        """
        Get Morgan fingerprint environment bits for a molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()
        
        # Get fingerprint and bit information
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits, bitInfo=bit_info
        )
        
        # Record substructure meaning for each bit (for interpretation)
        for bit, info_list in bit_info.items():
            if bit not in self.bit_meanings:
                # Get the atom environment corresponding to this bit
                if info_list:
                    atom_idx, radius = info_list[0]
                    env = self._get_environment_smiles(mol, atom_idx, radius)
                    self.bit_meanings[bit] = env
        
        return set(bit_info.keys())
    
    def _get_environment_smiles(self, mol, center_atom_idx, radius):
        """
        Get SMILES representation of specific atom environment
        """
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if submol:
                return Chem.MolToSmiles(submol)
            return f"Bit_{center_atom_idx}_r{radius}"
        except:
            return f"Bit_{center_atom_idx}_r{radius}"
    
    def compute_environment_frequencies(self, smiles_list, dataset_name=""):
        """
        Calculate frequency of each environment in dataset
        """
        print(f"Processing dataset {dataset_name}: {len(smiles_list)} molecules")
        
        all_environments = []
        valid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            if pd.isna(smiles):
                continue
                
            environments = self.get_morgan_environments(smiles)
            if environments:
                all_environments.extend(environments)
                valid_count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} molecules")
        
        # Calculate frequencies
        counter = Counter(all_environments)
        total_molecules = valid_count
        
        # Convert to frequency dictionary
        freq_dict = {}
        for bit, count in counter.items():
            freq_dict[bit] = count / total_molecules
        
        print(f"Valid molecules: {valid_count}, Unique environments: {len(freq_dict)}")
        return freq_dict, total_molecules
    
    def compare_datasets(self, smiles_list1, smiles_list2, 
                       dataset1_name="Dataset1", dataset2_name="Dataset2"):
        """
        Compare Morgan environment distributions between two datasets
        """
        # Calculate frequencies
        freq1, total1 = self.compute_environment_frequencies(smiles_list1, dataset1_name)
        freq2, total2 = self.compute_environment_frequencies(smiles_list2, dataset2_name)
        
        # Get all environment bits
        all_bits = set(freq1.keys()) | set(freq2.keys())
        print(f"\nTotal unique environments: {len(all_bits)}")
        
        # Create comparison DataFrame
        comparison_data = []
        for bit in all_bits:
            f1 = freq1.get(bit, 0)
            f2 = freq2.get(bit, 0)
            freq_diff = f1 - f2
            fold_change = f1 / f2 if f2 > 0 else np.inf
            
            comparison_data.append({
                'bit': bit,
                'environment': self.bit_meanings.get(bit, f"Bit_{bit}"),
                f'{dataset1_name}_freq': f1,
                f'{dataset2_name}_freq': f2,
                'frequency_diff': freq_diff,
                'fold_change': fold_change,
                'abs_freq_diff': abs(freq_diff)
            })
        
        df = pd.DataFrame(comparison_data)
        
        return df, freq1, freq2
    
    def get_significant_differences(self, comparison_df, min_freq=0.01, 
                                 min_fold_change=2.0, top_n=20):
        """
        Get environments with significant differences
        """
        # Filter low frequency environments
        filtered_df = comparison_df[
            (comparison_df.iloc[:, 2] >= min_freq) |  # dataset1 frequency
            (comparison_df.iloc[:, 3] >= min_freq)     # dataset2 frequency
        ]
        
        # Further filter by fold change
        significant_df = filtered_df[
            (filtered_df['fold_change'] >= min_fold_change) |
            (filtered_df['fold_change'] <= 1/min_fold_change)
        ]
        
        # Sort by frequency difference
        enriched_in_1 = significant_df.nlargest(top_n, 'frequency_diff')
        enriched_in_2 = significant_df.nsmallest(top_n, 'frequency_diff')
        
        return enriched_in_1, enriched_in_2
    
    def create_fingerprint_matrix(self, smiles_list1, smiles_list2, 
                               dataset1_name="Dataset1", dataset2_name="Dataset2"):
        """
        Create fingerprint matrix for dimensionality reduction
        """
        all_smiles = smiles_list1 + smiles_list2
        labels = [dataset1_name] * len(smiles_list1) + [dataset2_name] * len(smiles_list2)
        
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(all_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                fingerprints.append(fp)
                valid_indices.append(i)
        
        # Convert to numpy array
        fp_matrix = np.array(fingerprints)
        valid_labels = [labels[i] for i in valid_indices]
        
        return fp_matrix, valid_labels
    
    def visualize_tsne_comparison(self, smiles_list1, smiles_list2,
                               dataset1_name="Dataset1", dataset2_name="Dataset2",
                               output_dir="results/visualizations",
                               perplexity=30, random_state=42):
        """
        Perform t-SNE dimensionality reduction and save visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create fingerprint matrix
        fp_matrix, labels = self.create_fingerprint_matrix(
            smiles_list1, smiles_list2, dataset1_name, dataset2_name
        )
        
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        tsne_result = tsne.fit_transform(fp_matrix)
        
        # Create t-SNE visualization
        plt.figure(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        for i, label in enumerate([dataset1_name, dataset2_name]):
            mask = np.array(labels) == label
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       label=label, alpha=0.6, s=20, color=colors[i])
        
        plt.title(f't-SNE Visualization: {dataset1_name} vs {dataset2_name}\n(perplexity={perplexity})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_path = os.path.join(output_dir, f'tsne_comparison_{dataset1_name}_vs_{dataset2_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to: {output_path}")
        
        return tsne_result, labels

    def visualize_volcano_plot(self, comparison_df, dataset1_name="Dataset1", 
                            dataset2_name="Dataset2", output_dir="results/visualizations",
                            fold_change_threshold=2.0, freq_threshold=0.05):
        """
        Create volcano plot to visualize all environment bits
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for volcano plot
        volcano_data = comparison_df.copy()
        
        # Calculate log2 fold change, handling infinite values
        volcano_data['log2_fold_change'] = np.log2(
            volcano_data['fold_change'].replace([np.inf, -np.inf], np.nan)
        )
        
        # Remove NaN values
        volcano_data = volcano_data.dropna(subset=['log2_fold_change'])
        
        # Use absolute frequency difference as significance measure
        significance = volcano_data['abs_freq_diff']
        
        # Create volcano plot
        plt.figure(figsize=(12, 8))
        
        # Color points based on significance and fold change
        colors = []
        for _, row in volcano_data.iterrows():
            if (row['abs_freq_diff'] > freq_threshold and 
                abs(row['log2_fold_change']) > np.log2(fold_change_threshold)):
                if row['log2_fold_change'] > 0:
                    colors.append('red')  # Enriched in dataset1
                else:
                    colors.append('blue')  # Enriched in dataset2
            else:
                colors.append('gray')  # Not significant
        
        # Create scatter plot
        scatter = plt.scatter(volcano_data['log2_fold_change'], 
                            significance,
                            c=colors, 
                            alpha=0.6, 
                            s=30,
                            edgecolors='black', 
                            linewidth=0.5)
        
        # Add threshold lines
        plt.axvline(x=np.log2(fold_change_threshold), color='red', linestyle='--', 
                   alpha=0.8, linewidth=1, label=f'FC={fold_change_threshold}')
        plt.axvline(x=-np.log2(fold_change_threshold), color='red', linestyle='--', 
                   alpha=0.8, linewidth=1)
        plt.axhline(y=freq_threshold, color='green', linestyle='--', 
                   alpha=0.8, linewidth=1, label=f'Freq diff={freq_threshold}')
        
        plt.xlabel('log2(Fold Change)')
        plt.ylabel('Absolute Frequency Difference')
        plt.title(f'Volcano Plot: {dataset1_name} vs {dataset2_name}\n'
                 f'Total bits: {len(volcano_data)} | '
                 f'Significant: {sum(np.array(colors) != "gray")}')
        plt.grid(True, alpha=0.3)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Enriched in {dataset1_name}'),
            Patch(facecolor='blue', label=f'Enriched in {dataset2_name}'),
            Patch(facecolor='gray', label='Not significant'),
            Patch(facecolor='white', label=f'FC threshold: {fold_change_threshold}'),
            Patch(facecolor='white', label=f'Freq threshold: {freq_threshold}')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Annotate top significant points
        significant_mask = (volcano_data['abs_freq_diff'] > freq_threshold) & \
                         (volcano_data['log2_fold_change'].abs() > np.log2(fold_change_threshold))
        
        significant_points = volcano_data[significant_mask]
        if len(significant_points) > 0:
            # Get top 10 most significant points
            top_significant = significant_points.nlargest(10, 'abs_freq_diff')
            for _, row in top_significant.iterrows():
                plt.annotate(f"Bit {int(row['bit'])}", 
                            (row['log2_fold_change'], row['abs_freq_diff']),
                            xytext=(5, 5), 
                            textcoords='offset points', 
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'volcano_plot_{dataset1_name}_vs_{dataset2_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Volcano plot saved to: {output_path}")
        
        # Print summary statistics
        total_bits = len(volcano_data)
        sig_bits = len(significant_points)
        enriched_in_1 = len(significant_points[significant_points['log2_fold_change'] > 0])
        enriched_in_2 = len(significant_points[significant_points['log2_fold_change'] < 0])
        
        print(f"Volcano plot summary:")
        print(f"  Total environment bits: {total_bits}")
        print(f"  Significant bits: {sig_bits} ({sig_bits/total_bits*100:.1f}%)")
        print(f"  Enriched in {dataset1_name}: {enriched_in_1}")
        print(f"  Enriched in {dataset2_name}: {enriched_in_2}")
        
        return significant_points
    
    def visualize_frequency_comparison(self, comparison_df, dataset1_name="Dataset1", 
                                    dataset2_name="Dataset2", top_n=50,
                                    output_dir="results/visualizations"):
        """
        Visualize frequency comparison and save plot
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get enrichment results
        enriched_1, enriched_2 = self.get_significant_differences(
            comparison_df, top_n=top_n
        )
        
        # Combine data for plotting
        plot_data = pd.concat([enriched_1, enriched_2])
        
        # Create frequency comparison plot
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(plot_data))
        width = 0.35
        
        plt.bar(x_pos - width/2, plot_data.iloc[:, 2], width, 
                label=dataset1_name, alpha=0.7, color='#1f77b4')
        plt.bar(x_pos + width/2, plot_data.iloc[:, 3], width, 
                label=dataset2_name, alpha=0.7, color='#ff7f0e')
        
        plt.xlabel('Environment Features')
        plt.ylabel('Frequency')
        plt.title(f'Morgan Environment Frequency Comparison: {dataset1_name} vs {dataset2_name}')
        plt.legend()
        plt.xticks(x_pos, [f"Bit {row['bit']}" for _, row in plot_data.iterrows()], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'frequency_comparison_{dataset1_name}_vs_{dataset2_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Frequency comparison saved to: {output_path}")
        
        return plot_data

def load_and_sample_csv(csv_file, smiles_col='smiles', max_molecules=10000, random_state=42):
    """
    Load CSV file and randomly sample molecules
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Check if smiles column exists
    if smiles_col not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Column '{smiles_col}' not found. Available columns: {available_cols}")
    
    # Remove rows with missing SMILES
    df_clean = df.dropna(subset=[smiles_col])
    
    # Sample molecules
    if len(df_clean) > max_molecules:
        print(f"  Sampling {max_molecules} molecules from {len(df_clean)} total molecules")
        df_sampled = df_clean.sample(n=max_molecules, random_state=random_state)
    else:
        print(f"  Using all {len(df_clean)} molecules (less than max_molecules)")
        df_sampled = df_clean
    
    smiles_list = df_sampled[smiles_col].tolist()
    print(f"  Final sample size: {len(smiles_list)} molecules")
    
    return smiles_list

# Usage example
def analyze_morgan_environments(csv_file1, csv_file2, smiles_col='smiles',
                             dataset1_name="Dataset1", dataset2_name="Dataset2",
                             output_dir="results/visualizations",
                             tsne_perplexity=30,
                             volcano_fc_threshold=2.0,
                             volcano_freq_threshold=0.05,
                             max_molecules_per_dataset=10000,
                             random_state=42):
    """
    Complete analysis pipeline with random sampling
    """
    # Load and sample data
    smiles1 = load_and_sample_csv(csv_file1, smiles_col, max_molecules_per_dataset, random_state)
    smiles2 = load_and_sample_csv(csv_file2, smiles_col, max_molecules_per_dataset, random_state)
    
    print(f"\nFinal dataset sizes:")
    print(f"{dataset1_name}: {len(smiles1)} molecules")
    print(f"{dataset2_name}: {len(smiles2)} molecules")
    
    # Initialize comparator
    comparator = MorganEnvironmentComparator(radius=2, n_bits=2048)
    
    # Perform comparison
    comparison_df, freq1, freq2 = comparator.compare_datasets(
        smiles1, smiles2, dataset1_name, dataset2_name
    )
    
    # Get significant differences
    enriched_1, enriched_2 = comparator.get_significant_differences(comparison_df)
    
    # Print results
    print(f"\nEnvironments enriched in {dataset1_name} (Top 10):")
    for _, row in enriched_1.head(10).iterrows():
        print(f"  Bit {row['bit']}: {row['environment']}")
        print(f"    Frequency: {dataset1_name}={row[dataset1_name + '_freq']:.3f}, "
              f"{dataset2_name}={row[dataset2_name + '_freq']:.3f}, "
              f"Fold Change={row['fold_change']:.2f}")
    
    print(f"\nEnvironments enriched in {dataset2_name} (Top 10):")
    for _, row in enriched_2.head(10).iterrows():
        print(f"  Bit {row['bit']}: {row['environment']}")
        print(f"    Frequency: {dataset1_name}={row[dataset1_name + '_freq']:.3f}, "
              f"{dataset2_name}={row[dataset2_name + '_freq']:.3f}, "
              f"Fold Change={1/row['fold_change']:.2f}")
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    
    # t-SNE visualization
    tsne_result, labels = comparator.visualize_tsne_comparison(
        smiles1, smiles2, dataset1_name, dataset2_name, output_dir, perplexity=tsne_perplexity
    )
    
    # Frequency comparison visualization
    plot_data = comparator.visualize_frequency_comparison(
        comparison_df, dataset1_name, dataset2_name, output_dir=output_dir
    )
    
    # Volcano plot visualization
    significant_points = comparator.visualize_volcano_plot(
        comparison_df, dataset1_name, dataset2_name, output_dir,
        fold_change_threshold=volcano_fc_threshold,
        freq_threshold=volcano_freq_threshold
    )
    
    # Save comparison results to CSV
    results_csv_path = os.path.join(output_dir, f'comparison_results_{dataset1_name}_vs_{dataset2_name}.csv')
    comparison_df.to_csv(results_csv_path, index=False)
    print(f"Comparison results saved to: {results_csv_path}")
    
    return comparison_df, enriched_1, enriched_2, comparator, significant_points

# Run analysis
if __name__ == "__main__":
    results = analyze_morgan_environments(
        # csv_file2='data/baseline_ligands/NPGPT.csv', 
        csv_file1='data/coconut_raw/coconut_10000.csv',
        csv_file2='results/FlowNP-kekulized.csv', 
        smiles_col='smiles',
        dataset1_name='coconut',
        dataset2_name='FlowNP',
        output_dir='results/visualizations',
        tsne_perplexity=30,
        volcano_fc_threshold=2.0,
        volcano_freq_threshold=0.05,
        max_molecules_per_dataset=1000,  # 设置最大分子数
        random_state=42  # 设置随机种子保证可重复性
    )