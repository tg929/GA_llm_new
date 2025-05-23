import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
# For SA score, RDKit's Contrib module is often used.
# Ensure sascorer.py is accessible or RDKit is built with it.
try:
    from rdkit.Chem import SA_Score
    # If the above fails, it might be that SA_Score is not directly under Chem
    # but rather in Contrib. Let's try to import sascorer directly if needed.
    # This assumes sascorer.py is in PYTHONPATH or rdkit.Contrib.SA_Score can be imported.
except ImportError:
    # A common way to include sascorer if not directly available
    try:
        from rdkit.Contrib import SA_Score as SA_Score_contrib
        SA_Score = SA_Score_contrib 
    except ImportError:
        print("Warning: SA_Score module not found. SA scores will not be calculated.")
        SA_Score = None

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
    """Calculates QED for a list of RDKit Mol objects."""
    qed_scores = []
    if not mols:
        return qed_scores
    for mol in mols:
        try:
            qed_scores.append(QED.qed(mol))
        except Exception as e:
            # print(f"Warning: Could not calculate QED for a molecule. Error: {e}")
            pass # Can add more specific error handling or logging
    return qed_scores

def calculate_sa_scores(mols):
    """Calculates SA scores for a list of RDKit Mol objects."""
    sa_scores = []
    if not SA_Score or not mols:
        if not SA_Score:
            print("SA_Score module not available, skipping SA score calculation.")
        return sa_scores
        
    for mol in mols:
        try:
            # The SA_Score module usually has a function like 'calculateScore' or similar
            # This depends on the exact version/implementation of sascorer being used
            # A common pattern is:
            sa_scores.append(SA_Score.calculateScore(mol))
        except Exception as e:
            # print(f"Warning: Could not calculate SA score for a molecule. Error: {e}")
            pass
    return sa_scores

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
    results = (
        f"Metrics for Population: {os.path.basename(args.current_population_docked_file)}
"
        f"--------------------------------------------------
"
        f"Total molecules processed: {len(current_smiles_list)}
"
        f"Valid RDKit molecules for properties: {len(all_mols)}
"
        f"Molecules with docking scores: {len(docking_scores)}
"
        f"--------------------------------------------------
"
        f"Docking Score - Top 1: {top1_score:.4f if not np.isnan(top1_score) else 'N/A'}
"
        f"Docking Score - Top 10 Mean: {top10_mean_score:.4f if not np.isnan(top10_mean_score) else 'N/A'}
"
        f"Docking Score - Top 100 Mean: {top100_mean_score:.4f if not np.isnan(top100_mean_score) else 'N/A'}
"
        f"--------------------------------------------------
"
        f"Novelty (vs {os.path.basename(args.initial_population_file)}): {novelty:.4f}
"
        f"Diversity (Internal): {diversity:.4f}
"
        f"--------------------------------------------------
"
        f"QED - Mean: {mean_qed:.4f if not np.isnan(mean_qed) else 'N/A'}
"
        f"SA Score - Mean: {mean_sa:.4f if not np.isnan(mean_sa) else 'N/A'}
"
        f"--------------------------------------------------
"
    )
    
    print("
Calculation Results:")
    print(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")

if __name__ == "__main__":
    main()
