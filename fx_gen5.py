import sys
import os
from tdc import Evaluator, Oracle
import numpy as np

def init_evaluators():
    div_evaluator = Evaluator(name='Diversity')
    nov_evaluator = Evaluator(name='Novelty')
    qed_evaluator = Oracle(name='qed')
    sa_evaluator = Oracle(name='sa')
    return div_evaluator, nov_evaluator, qed_evaluator, sa_evaluator

def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    if len(smiles_list) == 0:
        return {
            'diversity': 0.0,
            'novelty': 0.0,
            'avg_qed': 0.0,
            'avg_sa': 0.0,
            'num_valid': 0
        }
    diversity = div_eval(smiles_list) if len(smiles_list) >= 2 else 0.0
    try:
        novelty = nov_eval(smiles_list, ref_smiles)
    except ZeroDivisionError:
        novelty = 0.0
    results = {
        'diversity': diversity,
        'novelty': novelty,
        'avg_qed': np.mean([qed_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'avg_sa': np.mean([sa_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'num_valid': len(smiles_list)
    }
    return results

if __name__ == "__main__":
    # 这里假设你有两个文件：一个是参考集（训练集），一个是待评估种群
    ref_file = "datasets/source_compounds/naphthalene_smiles.smi"
    pop_file = "1.smi"
    with open(ref_file) as f:
        ref_smiles = [line.strip() for line in f if line.strip()]
    with open(pop_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    metrics = evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles)
    print(metrics)
