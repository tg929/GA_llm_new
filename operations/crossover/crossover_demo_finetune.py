import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from tdc import Evaluator, Oracle  
import random
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import logging
import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge 
import autogrow.operators.crossover.execute_crossover as execute_crossover
import autogrow.operators.filter.execute_filters as Filter

# 配置日志
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "crossover.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("crossover")

# 初始化评估器
def init_evaluators():
    try:
        div_evaluator = Evaluator(name='Diversity')
        nov_evaluator = Evaluator(name='Novelty') 
        qed_evaluator = Oracle(name='qed')
        sa_evaluator = Oracle(name='sa')
        return div_evaluator, nov_evaluator, qed_evaluator, sa_evaluator
    except ImportError:
        print("请先安装TDC包:pip install tdc")
        exit(1)

# 评估种群函数
def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    # 添加空列表保护
    if len(smiles_list) == 0:
        return {
            'diversity': 0.0,
            'novelty': 0.0,
            'avg_qed': 0.0,
            'avg_sa': 0.0,
            'num_valid': 0
        }
        
    # 计算多样性时需要至少2个样本
    diversity = div_eval(smiles_list) if len(smiles_list)>=2 else 0.0
    
    # 计算新颖性时处理分母为零的情况
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

def main():
    parser = argparse.ArgumentParser(description='改进的GA交叉参数')
    parser.add_argument("--source_compound_file", "-s", type=str, required=True,
                      help="源化合物文件路径")
    parser.add_argument("--llm_generation_file", "-l", type=str, 
                      default=os.path.join(PROJECT_ROOT, "fragment_GPT/output/test0/crossovered0_frags_new_0.smi"),
                      help="LLM生成分子文件路径")
    parser.add_argument("--output_file", "-o", type=str, 
                      default=os.path.join(PROJECT_ROOT, "output/generation_crossover_0.smi"),
                      help="输出文件路径")
    parser.add_argument("--crossover_rate", type=float, default=0.8,
                      help="交叉率")
    parser.add_argument("--crossover_attempts", type=int, default=1,
                      help="交叉尝试次数")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "output"),
                      help="输出目录")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info("开始交叉操作")
    
    # 加载初始数据集
    base_smiles = []
    with open(args.source_compound_file, 'r') as f:
        base_smiles = [line.split()[0].strip() for line in f]
        logger.info(f"加载初始分子数量: {len(base_smiles)}")
    
    # 加载LLM生成数据
    with open(args.llm_generation_file, 'r') as f:
        base_smiles_tol = base_smiles + [line.strip() for line in f if line.strip()]
        logger.info(f"合并LLM生成分子后总数: {len(base_smiles_tol)}")
    
    # 合并初始种群
    initial_population = list(base_smiles_tol)
    initial_population_0 = list(base_smiles)
    
    # 初始化评估器
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    
    # 评估初始种群
    logger.info("评估初始种群")
    initial_metrics = evaluate_population(initial_population, div_eval, nov_eval, 
                                        qed_eval, sa_eval, base_smiles)
    logger.info(f"初始种群评估结果:\n{initial_metrics}")
    
    # 交叉参数配置
    vars = {
        'min_atom_match_mcs': 4,
        'max_time_mcs_prescreen': 1,
        'max_time_mcs_thorough': 1,
        'protanate_step': True,
        'number_of_crossovers': args.crossover_attempts,
        'filter_object_dict': {},
        'max_variants_per_compound': 1,
        'debug_mode': False,
        'gypsum_timeout_limit': 120.0,
        'gypsum_thoroughness': 3
    }
    
    # 执行交叉操作
    logger.info(f"开始交叉操作，本轮目标生成 {args.crossover_attempts} 个新分子")
    crossed_population = []
    attempts = 0
    while len(crossed_population) < args.crossover_attempts:
        parent1, parent2 = random.sample(initial_population, 2)
        try:
            mol1 = execute_crossover.convert_mol_from_smiles(parent1)
            mol2 = execute_crossover.convert_mol_from_smiles(parent2)
            if mol1 is None or mol2 is None:
                continue
        except:
            continue
        mcs_result = execute_crossover.test_for_mcs(vars, mol1, mol2)
        if mcs_result is None:
            continue
        ligand_new_smiles = None
        for attempt in range(3):
            ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
            if ligand_new_smiles is not None:
                break
        if ligand_new_smiles is None:
            continue
        if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
            crossed_population.append(ligand_new_smiles)
        attempts += 1
    logger.info(f"本轮交叉实际生成 {len(crossed_population)} 个新分子，尝试次数: {attempts}")
    
    # 保存新生成的交叉分子
    new_crossed_file = os.path.join(args.output_dir, "generation_0_crossed_new.smi")
    with open(new_crossed_file, 'w') as f:
        for smi in crossed_population:
            f.write(f"{smi}\n")
    logger.info(f"新生成的交叉分子已保存至: {new_crossed_file}")
    
    # 评估新种群
    logger.info("评估交叉后的新种群")
    new_population = initial_population + crossed_population
    
    final_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                      qed_eval, sa_eval, base_smiles)
    logger.info(f"交叉后整个种群评估结果:\n{final_metrics}")
    
    crossed_metrics = evaluate_population(crossed_population, div_eval, nov_eval,
                                        qed_eval, sa_eval, base_smiles)
    logger.info(f"交叉产生的新分子评估结果:\n{crossed_metrics}")
    
    # 保存最终结果
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")
    logger.info(f"最终结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main() 