import argparse
import os
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 添加项目根目录到路径，方便导入其他模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 完全重写SA Score导入部分
# RDKit中的SA Score计算通常位于rdkit.Contrib.SA_score.sascorer模块
SA_SCORE_CALCULATOR = None
try:
    # 尝试1：常规路径
    from rdkit.Contrib.SA_score import sascorer
    SA_SCORE_CALCULATOR = sascorer.calculateScore
    print("Successfully imported SA Score calculator from rdkit.Contrib.SA_score.sascorer")
except ImportError:
    try:
        # 尝试2：直接从sascorer模块导入
        import sascorer
        SA_SCORE_CALCULATOR = sascorer.calculateScore
        print("Successfully imported SA Score calculator from sascorer module")
    except ImportError:
        try:
            # 尝试3：从rdkit.Chem导入(某些版本可能放在这里)
            from rdkit.Chem import sascorer
            SA_SCORE_CALCULATOR = sascorer.calculateScore
            print("Successfully imported SA Score calculator from rdkit.Chem.sascorer")
        except ImportError:
            print("Warning: SA_Score module not found. SA scores will not be calculated.")
            SA_SCORE_CALCULATOR = None

# 导入已有的SA Score计算模块
try:
    # 首先尝试导入项目中现有的SA Score计算模块
    from fragment_GPT.utils.chem_utils import get_sa, get_qed
    print("成功从fragment_GPT.utils.chem_utils导入SA和QED计算函数")
    # 为了保持1-10的SA分数范围，我们需要逆转get_sa函数的计算
    def calculate_sa_original(mol):
        # get_sa返回(10 - sascorer.calculateScore(mol)) / 9，是0-1范围
        # 我们需要的是原始的sascorer.calculateScore(mol)，是1-10范围
        sa_normalized = get_sa(mol)  # 0-1范围的值
        sa_original = 10 - (sa_normalized * 9)  # 转回1-10范围
        return sa_original
except ImportError:
    print("尝试导入fragment_GPT.utils.chem_utils失败，使用备用方法")
    # 备用方法：直接导入sascorer
    try:
        # 尝试1：项目根目录下的utils
        sys.path.append(os.path.join(PROJECT_ROOT, 'fragment_GPT/utils'))
        import sascorer
        def calculate_sa_original(mol):
            return sascorer.calculateScore(mol)
        print("成功从项目的sascorer模块导入SA计算函数")
    except ImportError:
        try:
            # 尝试2：RDKit的SA Score
            from rdkit.Contrib.SA_score import sascorer
            def calculate_sa_original(mol):
                return sascorer.calculateScore(mol)
            print("成功从rdkit.Contrib.SA_score导入SA计算函数")
        except ImportError:
            print("警告：无法导入任何SA Score计算模块，SA分数将不可用")
            calculate_sa_original = None

def load_smiles_from_file(filepath):
    """Loads SMILES from a file, one SMILES per line."""
    smiles_list = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                smiles = line.strip().split()[0] # Assuming SMILES is the first part if line has more
                if smiles:
                    smiles_list.append(smiles)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return []
    return smiles_list

def load_smiles_and_scores_from_file(filepath):
    """Loads SMILES and scores from a file."""
    molecules = []
    scores = []
    smiles_list = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles = parts[0]
                    try:
                        score = float(parts[1])
                        molecules.append(smiles)
                        scores.append(score)
                        smiles_list.append(smiles)
                    except ValueError:
                        print(f"Warning: Could not parse score for SMILES: {smiles}")
                elif len(parts) == 1: # If only SMILES is present, no score
                    smiles_list.append(parts[0])


    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
    return smiles_list, molecules, scores


def get_rdkit_mols(smiles_list):
    """Converts a list of SMILES to RDKit Mol objects."""
    mols = []
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            valid_smiles.append(s)
        else:
            print(f"Warning: Could not parse SMILES: {s}")
    return mols, valid_smiles

def calculate_docking_stats(scores):
    """Calculates Top-1, Top-10 mean, Top-100 mean docking scores."""
    if not scores:
        return None, None, None

    sorted_scores = sorted(scores) # Docking scores, lower is better

    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else np.nan
    
    top10_scores = sorted_scores[:10]
    top10_mean = np.mean(top10_scores) if top10_scores else np.nan

    top100_scores = sorted_scores[:100]
    top100_mean = np.mean(top100_scores) if top100_scores else np.nan
    
    return top1_score, top10_mean, top100_mean

def calculate_novelty(current_smiles, initial_smiles_path):
    """Calculates novelty against an initial set of SMILES."""
    if not current_smiles:
        return 0.0

    initial_smiles = load_smiles_from_file(initial_smiles_path)
    if not initial_smiles:
        print("Warning: Initial population file is empty or not found. Novelty will be 100%.")
        return 1.0
        
    set_initial_smiles = set(initial_smiles)
    set_current_smiles = set(current_smiles)
    
    new_molecules = set_current_smiles - set_initial_smiles
    
    novelty = len(new_molecules) / len(set_current_smiles) if len(set_current_smiles) > 0 else 0.0
    return novelty

def calculate_diversity(mols):
    """Calculates internal diversity based on Morgan fingerprints and Tanimoto similarity."""
    if len(mols) < 2:
        return 0.0 # Or 1.0, or undefined. Typically 0 if no pairs to compare.

    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    
    sum_similarity = 0
    num_pairs = 0
    
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sum_similarity += similarity
            num_pairs += 1
            
    if num_pairs == 0: # Should not happen if len(mols) >= 2
        return 0.0 
        
    average_similarity = sum_similarity / num_pairs
    diversity = 1.0 - average_similarity
    return diversity

def calculate_qed_scores(mols):
    """Calculates QED (0-1范围) for a list of RDKit Mol objects."""
    qed_scores = []
    if not mols:
        return qed_scores
    for mol in mols:
        try:
            # 直接使用RDKit的QED计算函数，结果范围是0-1
            qed_scores.append(QED.qed(mol))
        except Exception as e:
            print(f"Warning: Could not calculate QED for a molecule. Error: {str(e)}")
    
    print(f"Successfully calculated QED scores for {len(qed_scores)} out of {len(mols)} molecules.")
    return qed_scores

def calculate_sa_scores(mols):
    """Calculates SA scores (1-10范围，值越低越好) for a list of RDKit Mol objects."""
    sa_scores = []
    if not calculate_sa_original or not mols:
        if not calculate_sa_original:
            print("SA Score calculator function not available, skipping SA score calculation.")
        return sa_scores
        
    for mol in mols:
        try:
            # 使用我们定义的calculate_sa_original函数，确保结果在1-10范围内
            sa_score = calculate_sa_original(mol)
            sa_scores.append(sa_score)
        except Exception as e:
            print(f"Warning: Could not calculate SA score for a molecule. Error: {str(e)}")
    
    print(f"Successfully calculated SA scores for {len(sa_scores)} out of {len(mols)} molecules.")
    return sa_scores

def print_calculation_results(results):
    """打印计算结果，避免格式问题"""
    print("Calculation Results:")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generation of molecules.")
    parser.add_argument("--current_population_docked_file", type=str, required=True,
                        help="Path to the SMILES file of the current population with docking scores (SMILES score per line).")
    parser.add_argument("--initial_population_file", type=str, required=True,
                        help="Path to the SMILES file of the initial population (for novelty calculation).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file to save calculated metrics (e.g., results.txt or results.csv).")
    
    args = parser.parse_args()

    print(f"Processing population file: {args.current_population_docked_file}")
    print(f"Using initial population for novelty: {args.initial_population_file}")
    print(f"Saving results to: {args.output_file}")

    # Load current population SMILES and scores
    current_smiles_list, scored_molecules_smiles, docking_scores = load_smiles_and_scores_from_file(args.current_population_docked_file)
    
    if not current_smiles_list:
        print("No SMILES found in the current population file. Exiting.")
        return

    # Convert SMILES to RDKit Mol objects for QED, SA, Diversity
    # Use all SMILES from the docked file for these chemical property calculations
    all_mols, valid_smiles_for_props = get_rdkit_mols(current_smiles_list)

    # 1. Docking Score Metrics
    top1_score, top10_mean_score, top100_mean_score = calculate_docking_stats(docking_scores)
    
    # 2. Novelty
    # Use all unique SMILES from the current docked population for novelty calculation
    novelty = calculate_novelty(list(set(current_smiles_list)), args.initial_population_file)
    
    # 3. Diversity (calculated on valid RDKit molecules from the current population)
    diversity = calculate_diversity(all_mols)
    
    # 4. QED Scores (calculated on valid RDKit molecules)
    qed_scores = calculate_qed_scores(all_mols)
    mean_qed = np.mean(qed_scores) if qed_scores else np.nan
    
    # 5. SA Scores (calculated on valid RDKit molecules)
    sa_scores = calculate_sa_scores(all_mols)
    mean_sa = np.mean(sa_scores) if sa_scores else np.nan

    # Prepare results
    # 安全地处理可能包含特殊字符的文件名
    population_filename = os.path.basename(args.current_population_docked_file)
    initial_population_filename = os.path.basename(args.initial_population_file)
    
    # 为了避免f-string格式化问题，使用传统的字符串格式化
    results = "Metrics for Population: {}\n".format(population_filename)
    results += "--------------------------------------------------\n"
    results += "Total molecules processed: {}\n".format(len(current_smiles_list))
    results += "Valid RDKit molecules for properties: {}\n".format(len(all_mols))
    results += "Molecules with docking scores: {}\n".format(len(docking_scores))
    results += "--------------------------------------------------\n"
    
    # 处理浮点数格式化，注意处理NaN情况
    if np.isnan(top1_score):
        results += "Docking Score - Top 1: N/A\n"
    else:
        results += "Docking Score - Top 1: {:.4f}\n".format(top1_score)
        
    if np.isnan(top10_mean_score):
        results += "Docking Score - Top 10 Mean: N/A\n"
    else:
        results += "Docking Score - Top 10 Mean: {:.4f}\n".format(top10_mean_score)
        
    if np.isnan(top100_mean_score):
        results += "Docking Score - Top 100 Mean: N/A\n"
    else:
        results += "Docking Score - Top 100 Mean: {:.4f}\n".format(top100_mean_score)
    
    results += "--------------------------------------------------\n"
    results += "Novelty (vs {}): {:.4f}\n".format(initial_population_filename, novelty)
    results += "Diversity (Internal): {:.4f}\n".format(diversity)
    results += "--------------------------------------------------\n"
    
    if np.isnan(mean_qed):
        results += "QED - Mean: N/A\n"
    else:
        results += "QED - Mean: {:.4f}\n".format(mean_qed)
        
    if np.isnan(mean_sa):
        results += "SA Score - Mean: N/A\n"
    else:
        results += "SA Score - Mean: {:.4f}\n".format(mean_sa)
    
    results += "--------------------------------------------------\n"
    
    print_calculation_results(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")

if __name__ == "__main__":
    main()
