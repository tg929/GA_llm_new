import sys
import os
PROJECT_ROOT = "/data1/tgy/GA_llm"
sys.path.insert(0, PROJECT_ROOT)
from tdc import Evaluator, Oracle  
import random
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import logging
import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter

# 配置日志
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "mutation.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("mutation")

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
    parser = argparse.ArgumentParser(description='改进的GA变异参数')
    parser.add_argument("--input_file", "-i", type=str, required=True,
                      help="输入分子文件路径")
    parser.add_argument("--llm_generation_file", "-l", type=str, 
                      default="/data1/tgy/GA_llm/fragment_GPT/output/test0/crossovered0_frags_new_0.smi",
                      help="LLM生成分子文件路径")
    parser.add_argument("--output_file", "-o", type=str, 
                      default="/data1/tgy/GA_llm/output/generation_0_mutationed.smi",
                      help="输出文件路径")
    parser.add_argument("--num_mutations", type=int, default=1,
                      help="变异尝试次数")
    parser.add_argument("--max_mutations", type=int, default=2,
                      help="每个父代最大变异尝试次数")
    parser.add_argument("--output_dir", type=str, default="/data1/tgy/GA_llm/output",
                      help="输出目录")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info("开始变异操作")
    
    # 加载初始数据集
    base_smiles = []
    with open(args.input_file, 'r') as f:
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
                                        qed_eval, sa_eval, base_smiles_tol)
    logger.info(f"初始种群评估结果:\n{initial_metrics}")
    
    # 变异参数配置
    vars = {
        'rxn_library': 'all_rxns',
        'rxn_library_file': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json',
        'function_group_library': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json',
        'complementary_mol_directory': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir',
        'filter_object_dict': {
            'Structure_check': lambda mol: mol is not None
        },
        'max_time_mcs_thorough': 1,
        'gypsum_thoroughness': 3
    }

    # 初始化变异器
    rxn_library_vars = [
        vars['rxn_library'],
        vars['rxn_library_file'],
        vars['function_group_library'],
        vars['complementary_mol_directory']
    ]
    
    logger.info(f"开始变异操作，本轮目标生成 {args.num_mutations} 个新分子")
    mutation_results = []
    attempts = 0
    while len(mutation_results) < args.num_mutations:
        parent = random.choice(initial_population)
        click_chem = SmileClickClass.SmilesClickChem(rxn_library_vars, [], vars['filter_object_dict'])
        for attempt in range(args.max_mutations):
            result = click_chem.run_smiles_click2(parent)
            if not result:
                continue
            valid_results = []
            for smi in result:
                try:
                    if all([check(smi) for check in vars['filter_object_dict'].values()]):
                        valid_results.append(smi)
                        break
                except:
                    continue
            if valid_results:
                chosen_smi = valid_results[0]
                if chosen_smi not in initial_population and chosen_smi not in mutation_results:
                    mutation_results.append(chosen_smi)
                    break
        attempts += 1
    logger.info(f"本轮突变实际生成 {len(mutation_results)} 个新分子，尝试次数: {attempts}")
    
    # 保存新生成的变异分子
    new_mutations_file = os.path.join(args.output_dir, "generation_0_mutation_new.smi")
    with open(new_mutations_file, 'w') as f:
        for smi in mutation_results:
            f.write(f"{smi}\n")
    logger.info(f"新生成的变异分子已保存至: {new_mutations_file}")
    
    # 评估新种群
    logger.info("评估变异后的新种群")
    new_population = initial_population + mutation_results
    
    final_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                      qed_eval, sa_eval, initial_population)
    logger.info(f"变异后整个种群评估结果:\n{final_metrics}")
    
    mutation_metrics = evaluate_population(mutation_results, div_eval, nov_eval,
                                         qed_eval, sa_eval, initial_population)
    logger.info(f"变异产生的新分子评估结果:\n{mutation_metrics}")
    
    # 保存最终结果
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")
    logger.info(f"最终结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main() 