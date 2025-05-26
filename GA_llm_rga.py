import argparse
import os
import numpy as np
import sys
import time
import logging
import subprocess
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def setup_logging(output_dir, generation_num):    
    log_file = os.path.join(output_dir, f"rga_evolution_{generation_num}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GA_llm_rga")

# 定义receptor_info_list，包含所有受体的信息---10种受体蛋白
receptor_info_list = [
    ('4r6e', './pdb/4r6e.pdb', -70.76, 21.82, 28.33, 15.0, 15.0, 15.0),
    ('3pbl', './pdb/3pbl.pdb', 9, 22.5, 26, 15, 15, 15),
    ('1iep', './pdb/1iep.pdb', 15.6138918, 53.38013513, 15.454837, 15, 15, 15),
    ('2rgp', './pdb/2rgp.pdb', 16.29212, 34.870818, 92.0353, 15, 15, 15),
    ('3eml', './pdb/3eml.pdb', -9.06363, -7.1446, 55.86259999, 15, 15, 15),
    ('3ny8', './pdb/3ny8.pdb', 2.2488, 4.68495, 51.39820000000001, 15, 15, 15),
    ('4rlu', './pdb/4rlu.pdb', -0.73599, 22.75547, -31.23689, 15, 15, 15),
    ('4unn', './pdb/4unn.pdb', 5.684346153, 18.1917, -7.3715, 15, 15, 15),
    ('5mo4', './pdb/5mo4.pdb', -44.901, 20.490354, 8.48335, 15, 15, 15),
    ('7l11', './pdb/7l11.pdb', -21.81481, -4.21606, -27.98378, 15, 15, 15),
]

def run_decompose(input_file, output_prefix, logger):    
    logger.info(f"开始分子分解: {input_file}")       
    decompose_dir = os.path.join(PROJECT_ROOT, "datasets/decompose/decompose_results")
    os.makedirs(decompose_dir, exist_ok=True)      
    output_file = os.path.join(decompose_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_dir, f"truncated_frags_{output_prefix}.smi")
    output_file4 = os.path.join(decompose_dir, f"decomposable_mols_{output_prefix}.smi")    
    decompose_script = os.path.join(PROJECT_ROOT, "datasets/decompose/demo_frags.py")
    cmd = [
        "python", decompose_script,
        "-i", input_file,
        "-o", output_file,
        "-o2", output_file2,
        "-o3", output_file3,
        "-o4", output_file4
    ]    
    process = subprocess.run(cmd, capture_output=True, text=True)      
    return output_file3
def run_gpt_generation(input_file, output_prefix, gen_num, logger):
    """运行GPT生成新分子"""
    logger.info(f"开始GPT生成: {input_file}")    
    output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
    os.makedirs(output_dir, exist_ok=True)
    fixed_output_file = os.path.join(output_dir, f"crossovered{gen_num}_frags_new_{gen_num}.smi")
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--output_file", fixed_output_file,  # 明确指定输出文件
        "--device", "0",
        "--seed", str(gen_num)
    ]    
    process = subprocess.run(cmd, capture_output=True, text=True)    
    return fixed_output_file   

def run_crossover(source_file, llm_file, output_file, gen_num, num_crossovers, logger):
    """运行分子交叉"""
    logger.info(f"开始分子交叉: 源文件 {source_file}, LLM生成文件 {llm_file}, 交叉生成新个体数目 {num_crossovers}")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        "python", crossover_script,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_attempts", str(num_crossovers)
    ]    
    process = subprocess.run(cmd, capture_output=True, text=True)    
    logger.info(f"分子交叉完成，生成文件: {output_file}")
    return output_file
def run_mutation(input_file, llm_file, output_file, num_mutations, logger):
    """运行分子变异"""
    logger.info(f"开始分子变异: 输入文件 {input_file}, LLM生成文件 {llm_file}, 变异生成新个体数目 {num_mutations}")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo_finetune.py")
    cmd = [
        "python", mutation_script,
        "--input_file", input_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--num_mutations", str(num_mutations)
    ]    
    process = subprocess.run(cmd, capture_output=True, text=True)        
    logger.info(f"分子变异完成，生成文件: {output_file}")
    return output_file

def run_filter(input_file, output_file, logger, args):
    """运行分子过滤"""
    logger.info(f"开始分子过滤: {input_file}")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    filter_params = [] 
    if args.LipinskiStrictFilter:
        filter_params.extend(["--LipinskiStrictFilter"])
    if args.LipinskiLenientFilter:
        filter_params.extend(["--LipinskiLenientFilter"])
    if args.GhoseFilter:
        filter_params.extend(["--GhoseFilter"])
    if args.GhoseModifiedFilter:
        filter_params.extend(["--GhoseModifiedFilter"])
    if args.MozziconacciFilter:
        filter_params.extend(["--MozziconacciFilter"])
    if args.VandeWaterbeemdFilter:
        filter_params.extend(["--VandeWaterbeemdFilter"])
    if args.PAINSFilter:
        filter_params.extend(["--PAINSFilter"])
    if args.NIHFilter:
        filter_params.extend(["--NIHFilter"])
    if args.BRENKFilter:
        filter_params.extend(["--BRENKFilter"])
    if args.No_Filters:
        filter_params.extend(["--No_Filters"])
    if args.alternative_filter:#自定义过滤器
        for filter_entry in args.alternative_filter:
            filter_params.extend(["--alternative_filter", filter_entry])    
    
    filter_script = os.path.join(PROJECT_ROOT, "operations/filter/filter_demo.py")
    cmd = [
        "python", filter_script,
        "--input", input_file,
        "--output", output_file
    ]    
    cmd.extend(filter_params)    
    logger.info(f"执行过滤命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)    
    logger.info(f"分子过滤完成，生成文件: {output_file}")
    return output_file

def run_multi_receptor_docking(input_file, output_dir, targets, logger):
    """运行多受体对接"""
    logger.info(f"开始多受体对接: {input_file}, 目标受体: {targets}")        
    os.makedirs(output_dir, exist_ok=True)    
    docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_utils_demo.py")
    mgltools_path = os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6")
    cmd = [
        "python", docking_script,
        "-i", input_file,
        "-o", output_dir,
        "-m", mgltools_path,
        "--targets"
    ]
    cmd.extend(targets)
    
    logger.info(f"执行对接命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)    
    # 检查综合得分文件是否生成
    combined_file = os.path.join(output_dir, "combined_docking_scores.smi")
    if not os.path.exists(combined_file):
        logger.error(f"找不到综合对接得分文件: {combined_file}")
        raise Exception("多受体对接失败，未生成综合得分文件")
        
    # 检查各个受体的对接结果文件
    docking_results = {}
    missing_targets = []
    docking_results_dir = os.path.join(output_dir, "docking_results")
    for target in targets:
        result_file = os.path.join(docking_results_dir, f"docked_{target}.smi")
        if os.path.exists(result_file):
            docking_results[target] = result_file
        else:
            missing_targets.append(target)
    
    if missing_targets:
        logger.warning(f"以下目标受体的对接结果文件未生成: {missing_targets}")
    
    logger.info(f"多受体对接完成，生成 {len(docking_results)} 个对接结果文件和1个综合得分文件")
    return docking_results, combined_file

def run_multi_receptor_docking_pipeline(input_file, output_file, targets, logger):
    """运行完整的多受体对接流程"""
    logger.info(f"开始完整的多受体对接流程: {input_file}")
    
    # 提取当前受体名称和代数信息
    target = targets[0] if targets else "unknown"
    
    # 验证输出路径是否与当前受体匹配
    if target not in output_file and len(targets) == 1:
        # 尝试从输入文件路径提取正确的输出路径模式
        input_dir = os.path.dirname(input_file)
        gen_info = os.path.basename(input_dir)  # 应该是类似 "generation_X" 的格式
        if "generation_" in gen_info:
            # 构建正确的输出路径
            correct_output_dir = input_dir  # 保持在同一代的目录中
            correct_output_filename = f"{gen_info}_docked.smi"
            correct_output_file = os.path.join(correct_output_dir, correct_output_filename)
            
            logger.warning(f"输出路径似乎不正确。应包含当前受体 '{target}'")
            logger.warning(f"原输出路径: {output_file}")
            logger.warning(f"已修正为: {correct_output_file}")
            
            output_file = correct_output_file
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    docking_dir = os.path.join(output_dir, "multi_receptor_docking")
    os.makedirs(docking_dir, exist_ok=True)
    
    # 运行多受体对接
    docking_results, combined_scores_file = run_multi_receptor_docking(input_file, docking_dir, targets, logger)
    
    if not docking_results:
        logger.error("未生成任何对接结果文件")
        raise Exception("多受体对接失败")
    
    # 复制综合得分文件到指定输出位置
    if os.path.exists(combined_scores_file):
        import shutil
        logger.info(f"将对接结果从 {combined_scores_file} 复制到 {output_file}")
        shutil.copy2(combined_scores_file, output_file)
        logger.info(f"已将综合得分文件复制到: {output_file} (受体: {target})")
    else:
        logger.error(f"综合得分文件不存在: {combined_scores_file}")
        raise Exception("多受体对接失败")
    
    logger.info(f"多受体对接流程完成，结果保存至: {output_file}")
    return output_file

def calculate_and_print_stats(docking_output, generation_num, logger):
    """计算并输出当前种群的分数统计信息"""
    # 读取对接结果文件中的分数
    molecules = []
    scores = []
    try:
        with open(docking_output, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        molecules.append(parts[0])
                        scores.append(float(parts[1]))
    except Exception as e:
        logger.error(f"读取对接结果文件失败: {str(e)}")
        return
    
    if not scores:
        logger.warning("对接结果中没有发现有效分数")
        return
    
    # 将分数从小到大排序（对接分数越小越好）
    sorted_scores = sorted(scores)
    
    # 计算统计信息
    mean_score = np.mean(sorted_scores)
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else None
    
    # 计算top10均值
    top10_scores = sorted_scores[:10] if len(sorted_scores) >= 10 else sorted_scores
    top10_mean = np.mean(top10_scores)
    # 计算top20均值
    top20_scores = sorted_scores[:20] if len(sorted_scores) >= 20 else sorted_scores
    top20_mean = np.mean(top20_scores)
    # 计算top50均值
    top50_scores = sorted_scores[:50] if len(sorted_scores) >= 50 else sorted_scores
    top50_mean = np.mean(top50_scores)
    # 计算top100均值
    top100_scores = sorted_scores[:100] if len(sorted_scores) >= 100 else sorted_scores
    top100_mean = np.mean(top100_scores)
    
    # 输出统计信息
    stats_message = (
        f"\n==================== Generation {generation_num} 统计信息 ====================\n"
        f"总分子数: {len(scores)}\n"
        f"所有分子得分均值: {mean_score:.4f}\n"
        f"Top1得分: {top1_score:.4f}\n"
        f"Top10得分均值: {top10_mean:.4f}\n"
        f"Top20得分均值: {top20_mean:.4f}\n"
        f"Top50得分均值: {top50_mean:.4f}\n"
        f"Top100得分均值: {top100_mean:.4f}\n"
        f"========================================================================\n"
    )
    
    # 输出到日志
    logger.info(stats_message)
    
    # 输出到控制台
    print(stats_message)

def select_seeds_for_next_generation(docking_output, seed_output, top_mols, diversity_mols, logger, elitism_mols=1, prev_elite_mols=None):
    """基于适应度和多样性选择种子分子，支持精英保留机制"""
    logger.info(f"开始选择种子分子: 从 {docking_output} 选择 {top_mols} 个适应度种子和 {diversity_mols} 个多样性种子，保留 {elitism_mols} 个精英分子")
    
    # 读取对接结果
    molecules = []
    scores = []
    try:
        with open(docking_output, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        molecules.append(parts[0])
                        scores.append(float(parts[1]))
    except Exception as e:
        logger.error(f"读取对接结果文件失败: {str(e)}")
        return None
    
    if not scores:
        logger.warning("对接结果中没有发现有效分数")
        return None
    
    # 按分数排序（对接分数越小越好）
    sorted_indices = np.argsort(scores)
    sorted_molecules = [molecules[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # 获取当前代得分最好的分子
    current_best_mol = sorted_molecules[0]
    current_best_score = sorted_scores[0]
    
    # 如果有上一代的精英分子，比较并选择最好的
    if prev_elite_mols:
        prev_best_mol = list(prev_elite_mols.keys())[0]
        prev_best_score = list(prev_elite_mols.values())[0]
        
        # 比较当前代最好分子和上一代精英分子
        if current_best_score < prev_best_score:
            # 如果当前代有更好的分子，使用当前代的
            new_elite_mols = {current_best_mol: current_best_score}
            logger.info(f"发现更好的分子，更新精英分子:")
            logger.info(f"上一代精英分子: {prev_best_mol} (得分: {prev_best_score})")
            logger.info(f"新的精英分子: {current_best_mol} (得分: {current_best_score})")
        else:
            # 如果上一代的精英分子更好，继续保留
            new_elite_mols = {prev_best_mol: prev_best_score}
            logger.info(f"保留上一代精英分子:")
            logger.info(f"当前代最好分子: {current_best_mol} (得分: {current_best_score})")
            logger.info(f"保留的精英分子: {prev_best_mol} (得分: {prev_best_score})")
    else:
        # 第一代，直接使用当前代最好的分子作为精英分子
        new_elite_mols = {current_best_mol: current_best_score}
        logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score})")
    
    # 从剩余分子中选择适应度种子（排除已选择的精英分子）
    remaining_molecules = [mol for mol in sorted_molecules if mol not in new_elite_mols]
    fitness_seeds = remaining_molecules[:top_mols]
    logger.info(f"已选择 {len(fitness_seeds)} 个适应度种子")
    
    # 选择多样性种子
    diversity_seeds = []
    remaining_molecules = remaining_molecules[top_mols:]
    
    if diversity_mols > 0 and remaining_molecules:
        # 使用简单的最大最小距离算法选择多样性分子
        selected_indices = []
        # 从剩余分子中随机选择第一个
        first_idx = np.random.randint(0, len(remaining_molecules))
        selected_indices.append(first_idx)
        diversity_seeds.append(remaining_molecules[first_idx])
        
        # 选择剩余的多样性分子
        for _ in range(min(diversity_mols - 1, len(remaining_molecules) - 1)):
            max_min_dist = -1
            best_idx = -1
            
            for i in range(len(remaining_molecules)):
                if i in selected_indices:
                    continue
                    
                # 计算与已选分子的最小距离
                min_dist = float('inf')
                for j in selected_indices:
                    # 使用简单的字符串相似度作为距离度量
                    dist = sum(a != b for a, b in zip(remaining_molecules[i], remaining_molecules[j]))
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                diversity_seeds.append(remaining_molecules[best_idx])
    
    logger.info(f"已选择 {len(diversity_seeds)} 个多样性种子")
    
    # 合并所有种子（精英分子 + 适应度种子 + 多样性种子）
    all_seeds = list(new_elite_mols.keys()) + fitness_seeds + diversity_seeds
    
    # 保存种子分子
    with open(seed_output, 'w') as f:
        for mol in all_seeds:
            f.write(f"{mol}\n")
    
    logger.info(f"种子选择完成，共选择 {len(all_seeds)} 个分子，保存至: {seed_output}")
    return seed_output, new_elite_mols

def limit_population_size(input_file, max_size, output_file=None):
    """限制种群大小，保留前max_size个分子"""
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r') as f:
        molecules = [line for line in f.readlines() if line.strip()]
    
    if len(molecules) <= max_size:
        return input_file
    
    limited_molecules = molecules[:max_size]
    
    with open(output_file, 'w') as f:
        for mol in limited_molecules:
            f.write(mol)
    
    return output_file

def run_evolution(generation_num, args, logger, prev_elite_mols=None):
    """执行一次完整的进化迭代，适用于单受体对接的流程"""
    target = args.targets[0]  # 获取当前处理的受体
    logger.info(f"开始第 {generation_num} 代进化 (受体: {target})")
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)

    # 0. 保持第0代不变
    if generation_num == 0:
        # 初代直接单受体对接
        current_population = args.initial_population
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        
        # 执行单受体对接 (使用修改后的多受体对接函数，但只传入一个受体)
        run_multi_receptor_docking_pipeline(current_population, docking_output, args.targets, logger)
        calculate_and_print_stats(docking_output, generation_num, logger)
        
        # 选种子
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation
        )

        return seed_output, new_elite_mols
    else:
        # 1. 读取上一代seed文件
        prev_seed_file = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_seeds.smi")
        logger.info(f"读取上一代种子文件: {prev_seed_file}")
        
        # 2. 分子分解
        decompose_output = run_decompose(prev_seed_file, f"gen{generation_num}_seed", logger)
        
        # 3. GPT生成新分子，并将这些新分子保留
        gpt_output = run_gpt_generation(decompose_output, f"gen{generation_num}_seed", generation_num, logger)
        logger.info(f"GPT生成的新分子将直接加入新种群")
        
        # 4. 种子之间进行交叉操作
        crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
        run_crossover(prev_seed_file, prev_seed_file, crossover_output, generation_num, args.num_crossovers, logger)
        logger.info(f"注意:交叉操作仅在种子之间进行,不使用GPT生成的分子")
        
        # 5. 变异操作：对种子进行变异
        mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
        run_mutation(prev_seed_file, prev_seed_file, mutation_output, args.num_mutations, logger)
        logger.info(f"注意:变异操作仅使用种子分子,不引入GPT生成的分子")
        
        # 6. 合并新种群：精英分子 + GPT生成的新分子 + 交叉产生的新分子 + 变异产生的新分子
        new_population_file = os.path.join(output_base, f"generation_{generation_num}_new_population.smi")
        with open(new_population_file, 'w') as fout:
            # 首先写入精英分子（如果有的话）
            if prev_elite_mols:
                for mol, score in prev_elite_mols.items():
                    fout.write(f"{mol}\n")
                logger.info(f"已将上一代精英分子 {list(prev_elite_mols.keys())[0]} (得分: {list(prev_elite_mols.values())[0]}) 加入新种群")
            
            # 写入GPT生成的新分子
            gpt_new_molecules = 0
            with open(gpt_output, 'r') as fin:
                lines = fin.readlines()
                for line in lines:
                    if line.strip():
                        fout.write(line)
                        gpt_new_molecules += 1
            logger.info(f"已将GPT生成的 {gpt_new_molecules} 个分子加入新种群")
            
            # 写入交叉和变异产生的新分子
            total_new_molecules = gpt_new_molecules
            for fname, operation in [(crossover_output, "交叉"), (mutation_output, "变异")]:
                with open(fname, 'r') as fin:
                    lines = fin.readlines()
                    new_mols_count = 0
                    for line in lines:
                        if line.strip():
                            fout.write(line)
                            new_mols_count += 1
                    total_new_molecules += new_mols_count
                    logger.info(f"已将{operation}产生的 {new_mols_count} 个分子加入新种群")
            
            logger.info(f"新种群总计 {total_new_molecules + (1 if prev_elite_mols else 0)} 个分子")
        
        # 7. 对新种群进行单受体对接 (使用修改后的多受体对接函数，但只传入一个受体)
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_multi_receptor_docking_pipeline(new_population_file, docking_output, args.targets, logger)
        calculate_and_print_stats(docking_output, generation_num, logger)
        
        # 8. 选择下一代种子
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation, prev_elite_mols
        )

        return seed_output, new_elite_mols

def run_evolution_for_target(target, args, generations):
    """为单个受体运行完整的进化过程"""
    # 为当前受体创建单独的输出目录
    target_output_dir = os.path.join(args.output_dir, f"target_{target}")
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 创建当前受体的参数副本，并修改输出目录和目标受体
    target_args = argparse.Namespace(**vars(args))
    target_args.output_dir = target_output_dir
    
    # 获取当前受体的信息（从receptor_info_list中）
    receptor_info = next((info for info in receptor_info_list if info[0] == target), None)
    if not receptor_info:
        print(f"错误: 未找到受体 {target} 的信息")
        return
    
    # 设置受体文件路径和对接盒子参数
    target_args.receptor_file = receptor_info[1]
    target_args.center_x = receptor_info[2]
    target_args.center_y = receptor_info[3]
    target_args.center_z = receptor_info[4]
    target_args.size_x = receptor_info[5]
    target_args.size_y = receptor_info[6]
    target_args.size_z = receptor_info[7]
    
    print(f"======== 开始针对受体 {target} 的进化过程 ========")
    
    # 执行多代进化
    logger = setup_logging(target_output_dir, 0)
    elite_mols = None
    
    try:
        logger.info(f"开始第0代进化 (对初始种群直接进行对接 - 目标受体: {target})")
        start_time = time.time()
        
        # 创建第0代输出目录
        gen0_output_dir = os.path.join(target_output_dir, "generation_0")
        os.makedirs(gen0_output_dir, exist_ok=True)
        
        # 对初始种群进行对接
        docking_output = os.path.join(gen0_output_dir, "generation_0_docked.smi")
        run_docking(target_args.initial_population, docking_output, target_args.receptor_file, 
                   target_args.mgltools_path, logger, target_args.number_of_processors, 
                   target_args.multithread_mode)
        
        # 计算统计信息
        calculate_and_print_stats(docking_output, 0, logger)
        
        # 选择种子分子
        diversity_mols = max(0, target_args.diversity_mols_to_seed_first_generation)
        seed_output = os.path.join(gen0_output_dir, "generation_0_seeds.smi")
        seed_output, elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, target_args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, target_args.elitism_mols_to_next_generation
        )
        
        end_time = time.time()
        logger.info(f"第0代完成,耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"第0代失败: {str(e)}")
        print(f"受体 {target} 的第0代进化失败: {str(e)}")
        return  # 如果第0代失败，跳过此受体的后续代
    
    # 执行后续代进化
    for gen in range(1, generations + 1):
        logger = setup_logging(target_output_dir, gen)
        try:
            logger.info(f"开始第 {gen} 代进化，目标受体: {target}")
            start_time = time.time()
            
            # 如果前一代种群存在且超过限制大小，先限制它
            if target_args.max_population > 0:
                prev_gen_file = os.path.join(target_output_dir, f"generation_{gen-1}", f"generation_{gen-1}_docked.smi")
                if os.path.exists(prev_gen_file):
                    with open(prev_gen_file, 'r') as f:
                        prev_count = sum(1 for line in f if line.strip())
                    if prev_count > target_args.max_population:
                        limit_population_size(prev_gen_file, target_args.max_population)
                        logger.info(f"第{gen-1}代种群已从{prev_count}限制为{target_args.max_population}")
            
            # 创建当前代输出目录
            gen_output_dir = os.path.join(target_output_dir, f"generation_{gen}")
            os.makedirs(gen_output_dir, exist_ok=True)
            
            # 1. 读取上一代seed文件
            prev_seed_file = os.path.join(target_output_dir, f"generation_{gen-1}", f"generation_{gen-1}_seeds.smi")
            logger.info(f"读取上一代种子文件: {prev_seed_file}")
            
            # 2. 分子分解
            decompose_output = run_decompose(prev_seed_file, f"gen{gen}_seed", logger)
            
            # 3. GPT生成新分子，并将这些新分子保留
            gpt_output = run_gpt_generation(decompose_output, f"gen{gen}_seed", gen, logger)
            logger.info(f"GPT生成的新分子将直接加入新种群")
            
            # 4. 种子之间进行交叉操作
            crossover_output = os.path.join(gen_output_dir, f"generation_{gen}_crossover.smi")
            run_crossover(prev_seed_file, prev_seed_file, crossover_output, gen, target_args.num_crossovers, logger)
            logger.info(f"注意:交叉操作仅在种子之间进行,不使用GPT生成的分子")
            
            # 5. 变异操作：对种子进行变异
            mutation_output = os.path.join(gen_output_dir, f"generation_{gen}_mutation.smi")
            run_mutation(prev_seed_file, prev_seed_file, mutation_output, target_args.num_mutations, logger)
            logger.info(f"注意:变异操作仅使用种子分子,不引入GPT生成的分子")
            
            # 6. 合并新种群：精英分子 + GPT生成的新分子 + 交叉产生的新分子 + 变异产生的新分子
            new_population_file = os.path.join(gen_output_dir, f"generation_{gen}_new_population.smi")
            with open(new_population_file, 'w') as fout:
                # 首先写入精英分子（如果有的话）
                if elite_mols:
                    for mol, score in elite_mols.items():
                        fout.write(f"{mol}\n")
                    logger.info(f"已将上一代精英分子 {list(elite_mols.keys())[0]} (得分: {list(elite_mols.values())[0]}) 加入新种群")
                
                # 写入GPT生成的新分子
                gpt_new_molecules = 0
                with open(gpt_output, 'r') as fin:
                    lines = fin.readlines()
                    for line in lines:
                        if line.strip():
                            fout.write(line)
                            gpt_new_molecules += 1
                logger.info(f"已将GPT生成的 {gpt_new_molecules} 个分子加入新种群")
                
                # 写入交叉和变异产生的新分子
                total_new_molecules = gpt_new_molecules
                for fname, operation in [(crossover_output, "交叉"), (mutation_output, "变异")]:
                    with open(fname, 'r') as fin:
                        lines = fin.readlines()
                        new_mols_count = 0
                        for line in lines:
                            if line.strip():
                                fout.write(line)
                                new_mols_count += 1
                        total_new_molecules += new_mols_count
                        logger.info(f"已将{operation}产生的 {new_mols_count} 个分子加入新种群")
                
                logger.info(f"新种群总计 {total_new_molecules + (1 if elite_mols else 0)} 个分子")
            
            # 7. 对新种群进行对接
            docking_output = os.path.join(gen_output_dir, f"generation_{gen}_docked.smi")
            run_docking(new_population_file, docking_output, target_args.receptor_file, 
                       target_args.mgltools_path, logger, target_args.number_of_processors, 
                       target_args.multithread_mode)
            calculate_and_print_stats(docking_output, gen, logger)
            
            # 8. 选择下一代种子
            diversity_mols = max(0, target_args.diversity_mols_to_seed_first_generation - 
                               (gen * target_args.diversity_seed_depreciation_per_gen))
            seed_output = os.path.join(gen_output_dir, f"generation_{gen}_seeds.smi")
            seed_output, elite_mols = select_seeds_for_next_generation(
                docking_output, seed_output, target_args.top_mols_to_seed_next_generation, 
                diversity_mols, logger, target_args.elitism_mols_to_next_generation, elite_mols
            )
            
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {seed_output}")
            
        except Exception as e:
            logger.error(f"第 {gen} 代进化失败: {str(e)}")
            print(f"受体 {target} 的第 {gen} 代进化失败: {str(e)}")
            break  # 如果某一代失败，跳过此受体的后续代
    
    print(f"======== 受体 {target} 的进化过程完成 ========")
    return target

def get_available_cpu_count():
    """获取当前系统可用的CPU核心数量"""
    try:
        # 获取CPU使用率小于80%的核心数量
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        available_cores = sum(1 for percent in cpu_percent if percent < 80)
        
        # 确保至少使用一个核心
        return max(1, available_cores)
    except Exception as e:
        # 如果无法获取CPU使用情况，默认使用全部核心
        print(f"无法获取CPU使用情况: {str(e)}，将使用全部核心")
        return os.cpu_count()

def run_docking(input_file, output_file, receptor_file, mgltools_path, logger, num_processors=1, multithread_mode="serial"):
    """运行分子对接，针对单个受体蛋白"""
    logger.info(f"开始分子对接: {input_file} 对接到 {receptor_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定处理器数量 - 如果为-1或大于可用CPU数量，则使用所有可用CPU
    available_cpus = multiprocessing.cpu_count()
    if num_processors == -1 or num_processors > available_cpus:
        num_processors = available_cpus
        logger.info(f"自动设置使用所有可用的CPU核心: {num_processors}")
    
    # 构建对接命令
    docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
    cmd = [
        "python", docking_script,
        "-i", input_file,
        "-r", receptor_file,
        "-o", output_file,
        "-m", mgltools_path,
        "--number_of_processors", str(num_processors),
        "--multithread_mode", multithread_mode
    ]
    
    logger.info(f"执行对接命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子对接失败: {process.stderr}")
        raise Exception("分子对接失败")
    
    logger.info(f"分子对接完成，生成文件: {output_file}")
    return output_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm_rga - 基于多受体对接的分子进化与生成流程')
    
    # 检查psutil库依赖
    try:
        import psutil
    except ImportError:
        print("警告: 未找到psutil库,无法检测CPU空闲核心数量")
        print("请使用 'pip install psutil' 安装此依赖，或直接指定--max_workers参数")
        print("程序将继续执行,但会使用全部可用CPU核心...\n")
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=5, 
                        help='进化代数(不包括第0代,总共会生成6代:generation_0到generation_5)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'output_rga'),
                        help='基础输出目录，每个受体会在此目录下创建子目录')
    parser.add_argument('--initial_population', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'datasets/source_compounds/naphthalene_smiles.smi'),
                        help='初始种群文件路径')
    
    # 对接参数
    parser.add_argument('--targets', nargs='+', 
                        default=['4r6e', '3pbl', '1iep', '2rgp', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11'], 
                        help='受体蛋白列表')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='是否并行处理不同受体的进化过程')
    parser.add_argument('--max_workers', type=int, default=-1,
                        help='并行处理时的最大进程数，默认为-1表示自动检测并使用所有空闲CPU核心')
    parser.add_argument('--mgltools_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'mgltools_x86_64Linux2_1.5.6'),
                        help='MGLTools安装路径')
    
    # 进化参数
    parser.add_argument('--num_crossovers', type=int, default=50,
                       help='每代通过交叉生成的新分子数量')
    parser.add_argument('--num_mutations', type=int, default=50,
                       help='每代通过变异生成的新分子数量')
    parser.add_argument('--max_population', type=int, default=0,
                       help='控制每代种群的最大数量,设置为0表示不限制(可能导致种群规模迅速增长）')
    
    # 种子选择参数
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=10,
                       help='每代基于适应度选择进入下一代的分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=10,
                       help='第0代基于多样性选择进入下一代的分子数量')
    parser.add_argument('--diversity_seed_depreciation_per_gen', type=int, default=2,
                       help='每代多样性种子数量的递减值')
    parser.add_argument('--elitism_mols_to_next_generation', type=int, default=1,
                       help='每代保留的精英分子数量，这些分子将直接进入下一代而不进行进化操作')
    
    # 并行处理参数
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于对接计算的处理器数量，设置为-1表示自动检测并使用所有可用CPU核心')
    parser.add_argument('--multithread_mode', default="multithreading",
                        choices=["mpi", "multithreading", "serial"],
                        help='多线程模式选择: mpi, multithreading, 或 serial。serial模式将忽略处理器数量设置,强制使用单处理器。')
    
    # 过滤器参数
    parser.add_argument('--LipinskiStrictFilter', action='store_true', default=False,
                        help='严格版Lipinski五规则过滤器,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。要求必须通过所有条件。')
    parser.add_argument('--LipinskiLenientFilter', action='store_true', default=False,
                        help='宽松版Lipinski五规则过滤,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。允许一个条件不满足。')
    parser.add_argument('--GhoseFilter', action='store_true', default=False,
                        help='Ghose药物相似性过滤器,通过分子量、logP和原子数量进行筛选。')
    parser.add_argument('--GhoseModifiedFilter', action='store_true', default=False,
                        help='修改版Ghose过滤器,将分子量上限从480Da放宽到500Da。设计用于与Lipinski过滤器配合使用。')
    parser.add_argument('--MozziconacciFilter', action='store_true', default=False,
                        help='Mozziconacci药物相似性过滤器,评估可旋转键、环、氧原子和卤素原子的数量。')
    parser.add_argument('--VandeWaterbeemdFilter', action='store_true', default=False,
                        help='筛选可能透过血脑屏障的药物，基于分子量和极性表面积(PSA)。')
    parser.add_argument('--PAINSFilter', action='store_true', default=False,
                        help='PAINS过滤器,用于过滤泛测试干扰化合物,使用子结构搜索。')
    parser.add_argument('--NIHFilter', action='store_true', default=False,
                        help='NIH过滤器,过滤含有不良功能团的分子，使用子结构搜索。')
    parser.add_argument('--BRENKFilter', action='store_true', default=False,
                        help='BRENK前导物相似性过滤器,排除常见假阳性分子。')
    parser.add_argument('--No_Filters', action='store_true', default=False,
                        help='设置为True时,不应用任何过滤器。')
    parser.add_argument('--alternative_filter', action='append',
                        help='添加自定义过滤器，需要提供列表格式：[[过滤器1名称, 过滤器1路径], [过滤器2名称, 过滤器2路径]]')
    
    args = parser.parse_args()
    
    # 创建基础输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果设置了种群大小限制，检查初始种群
    if args.max_population > 0:
        # 检查初始种群大小
        with open(args.initial_population, 'r') as f:
            initial_count = sum(1 for line in f if line.strip())
        if initial_count > args.max_population:
            limited_file = os.path.join(args.output_dir, "limited_initial_population.smi")
            args.initial_population = limit_population_size(args.initial_population, args.max_population, limited_file)
            print(f"初始种群已从{initial_count}限制为{args.max_population}")
    
    # 确定处理器数量
    max_workers = args.max_workers
    if max_workers == -1:
        # 使用自动检测的空闲CPU核心数量
        max_workers = get_available_cpu_count()
        print(f"自动检测到 {max_workers} 个空闲CPU核心，将全部用于并行处理")
    elif max_workers <= 0 and max_workers != -1:
        # 对于其他非法值，使用所有CPU核心
        max_workers = os.cpu_count()
        print(f"指定的核心数无效，将使用所有 {max_workers} 个CPU核心进行并行处理")
    
    # 为每个受体蛋白分别执行完整的进化过程
    if args.parallel:
        print(f"使用并行模式处理 {len(args.targets)} 个受体蛋白，最大进程数: {max_workers}")
        # 使用多进程并行处理不同受体
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 为每个受体提交一个任务
            futures = {executor.submit(run_evolution_for_target, target, args, args.generations): target 
                      for target in args.targets}
            
            # 等待所有任务完成
            for future in as_completed(futures):
                target = futures[future]
                try:
                    result = future.result()
                    print(f"受体 {target} 的进化过程已完成!")
                except Exception as e:
                    print(f"受体 {target} 的进化过程发生错误: {str(e)}")
    else:
        print(f"使用串行模式处理 {len(args.targets)} 个受体蛋白")
        # 串行处理不同受体
        for target in args.targets:
            run_evolution_for_target(target, args, args.generations)
    
    print("所有受体的进化过程已完成！")

if __name__ == "__main__":
    main()
