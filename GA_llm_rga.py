#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_llm_rga.py - 基于NSGA-II帕累托多目标优化的分子进化与生成流程

=============================================================================
程序功能：
=============================================================================
本程序实现了一个基于遗传算法的分子优化框架，采用NSGA-II帕累托算法进行多目标优化。
主要特性包括：

1. 多目标优化：同时优化对接分数、QED分数和SA分数
   - 对接分数：最小化（结合亲和力）
   - QED分数：最大化（药物相似性）  
   - SA分数：最小化（合成难度）

2. 多受体并行处理：支持对多个蛋白受体同时进行优化

3. 遗传算法操作：
   - 分子分解
   - GPT辅助生成
   - 交叉操作
   - 变异操作
   - 精英保留

4. NSGA-II帕累托选择：真正的多目标优化，找到帕累托最优解集

=============================================================================
与之前版本的主要改进：
=============================================================================
- 删除了基于综合评分Y的单目标优化方法
- 引入真正的NSGA-II帕累托多目标优化
- 调用专门的多目标选择脚本 operations/selecting/selecting_multi_demo.py
- 简化了参数配置，专注于多目标优化

=============================================================================
使用示例：
=============================================================================

# 基本使用 - 对单个受体进行优化
python GA_llm_rga.py --targets 4r6e --generations 5

# 多受体并行优化
python GA_llm_rga.py --targets 4r6e 3pbl 1iep --parallel --max_workers 4

# 自定义种子选择参数
python GA_llm_rga.py --targets 4r6e \\
    --top_mols_to_seed_next_generation 15 \\
    --diversity_mols_to_seed_first_generation 15 \\
    --generations 10

# 启用GPU串行模式(适用于显存不足的情况)
python GA_llm_rga.py --targets 4r6e --gpt_serial_mode

=============================================================================
"""

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
import threading

# 获取当前Python可执行文件路径
PYTHON_EXECUTABLE = sys.executable

# 尝试导入 psutil
_psutil_available = False
try:
    import psutil
    _psutil_available = True
except ImportError:
    print("警告: psutil库未找到或导入失败。CPU核心数检测可能不准确。请考虑运行 'pip install psutil' 来安装。")

# 全局GPT串行锁
_gpt_serial_lock = threading.Lock()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 定义receptor_info_list，包含所有受体的信息---10种受体蛋白
receptor_info_list = [
    ('4r6e', os.path.join(PROJECT_ROOT, 'pdb', '4r6e.pdb'), -70.76, 21.82, 28.33, 15.0, 15.0, 15.0),
    ('3pbl', os.path.join(PROJECT_ROOT, 'pdb', '3pbl.pdb'), 9, 22.5, 26, 15, 15, 15),
    ('1iep', os.path.join(PROJECT_ROOT, 'pdb', '1iep.pdb'), 15.6138918, 53.38013513, 15.454837, 15, 15, 15),
    ('2rgp', os.path.join(PROJECT_ROOT, 'pdb', '2rgp.pdb'), 16.29212, 34.870818, 92.0353, 15, 15, 15),
    ('3eml', os.path.join(PROJECT_ROOT, 'pdb', '3eml.pdb'), -9.06363, -7.1446, 55.86259999, 15, 15, 15),
    ('3ny8', os.path.join(PROJECT_ROOT, 'pdb', '3ny8.pdb'), 2.2488, 4.68495, 51.39820000000001, 15, 15, 15),
    ('4rlu', os.path.join(PROJECT_ROOT, 'pdb', '4rlu.pdb'), -0.73599, 22.75547, -31.23689, 15, 15, 15),
    ('4unn', os.path.join(PROJECT_ROOT, 'pdb', '4unn.pdb'), 5.684346153, 18.1917, -7.3715, 15, 15, 15),
    ('5mo4', os.path.join(PROJECT_ROOT, 'pdb', '5mo4.pdb'), -44.901, 20.490354, 8.48335, 15, 15, 15),
    ('7l11', os.path.join(PROJECT_ROOT, 'pdb', '7l11.pdb'), -21.81481, -4.21606, -27.98378, 15, 15, 15),
]

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

def run_decompose(input_file, output_prefix, logger, current_gen_output_dir):
    logger.info(f"开始分子分解: {input_file} (输出到: {current_gen_output_dir})")
    decompose_base_dir = os.path.join(current_gen_output_dir, "decompose_results")
    os.makedirs(decompose_base_dir, exist_ok=True)

    # 使用 output_prefix 来确保在同一代中，不同调用（如果发生）的文件名是唯一的
    # 例如，如果 output_prefix 是 "gen1_seed_target_4r6e"
    output_file = os.path.join(decompose_base_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_base_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_base_dir, f"truncated_frags_{output_prefix}.smi") # 这是通常被下游使用的文件
    output_file4 = os.path.join(decompose_base_dir, f"decomposable_mols_{output_prefix}.smi")

    decompose_script = os.path.join(PROJECT_ROOT, "datasets/decompose/demo_frags.py")
    cmd = [
        PYTHON_EXECUTABLE, decompose_script,
        "-i", input_file,
        "-o", output_file,
        "-o2", output_file2,
        "-o3", output_file3,
        "-o4", output_file4
    ]    
    logger.info(f"执行分子分解命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)      

    if process.returncode != 0:
        logger.error(f"分子分解失败。返回码: {process.returncode}")
        logger.error(f"Stdout: {process.stdout}")
        logger.error(f"Stderr: {process.stderr}")
        raise Exception(f"分子分解失败: {input_file}")
    else:
        logger.info(f"分子分解成功。主要输出文件: {output_file3}")

    if not os.path.exists(output_file3):
        logger.error(f"分子分解声称成功，但主要输出文件未找到: {output_file3}")
        raise Exception(f"分子分解后文件丢失: {output_file3}")

    return output_file3

def run_gpt_generation(input_file, output_prefix, gen_num, logger, current_gen_output_dir, gpt_serial_mode=False):
    """运行GPT生成新分子"""
    logger.info(f"开始GPT生成: {input_file} (输出到: {current_gen_output_dir}, 串行模式: {gpt_serial_mode})")
    
    # 如果启用GPT串行模式，获取全局锁
    if gpt_serial_mode:
        logger.info("GPT串行模式已启用,等待获取串行锁...")
        _gpt_serial_lock.acquire()
        logger.info("已获取GPT串行锁,开始执行...")
    
    try:
        # 为GPT生成创建一个专用的子目录，以避免与同一代数的其他文件冲突
        gpt_output_base_dir = os.path.join(current_gen_output_dir, "fragment_GPT_output")
        os.makedirs(gpt_output_base_dir, exist_ok=True)       
        # 默认输出位置：PROJECT_ROOT/fragment_GPT/output/
        # 输出文件名：crossovered0_frags_new_{seed}.smi (有效分子)
        # 输出文件名：crossovered0_fragsCom_new_{seed}.smi (所有生成内容)        
        # 创建更独特的seed以避免并行冲突
        # 使用当前时间戳的后几位 + gen_num + output_prefix的哈希值
        import hashlib
        import time
        timestamp_suffix = int(time.time() * 1000) % 10000  # 取时间戳的后4位
        prefix_hash = abs(hash(output_prefix)) % 1000  # 取output_prefix哈希的后3位
        unique_seed = int(f"{gen_num}{timestamp_suffix}{prefix_hash}")        
        # GPU设备分配策略：基于output_prefix选择GPU设备
        # 检查可用的GPU数量
        if gpt_serial_mode:
            # 串行模式下始终使用GPU 0
            gpu_id = 0
            logger.info(f"GPT串行模式:使用GPU 0")
        else:
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    # 基于output_prefix的哈希值分配GPU
                    gpu_id = abs(hash(output_prefix)) % gpu_count
                    logger.info(f"检测到 {gpu_count} 个GPU,为此任务分配GPU {gpu_id}")
                else:
                    gpu_id = 0
                    logger.info(f"只有1个GPU可用,使用GPU 0")
            except:
                gpu_id = 0
                logger.warning(f"无法检测GPU数量,默认使用GPU 0")
        
        logger.info(f"使用独特seed: {unique_seed} (基于gen_num={gen_num}, timestamp_suffix={timestamp_suffix}, prefix_hash={prefix_hash})")
        
        generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
        cmd = [
            PYTHON_EXECUTABLE, generate_script,
            "--input_file", input_file,
            "--device", str(gpu_id), # 使用分配的GPU设备
            "--seed", str(unique_seed) # 使用独特的seed
        ]    
        logger.info(f"执行GPT生成命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)    

        if process.returncode != 0:
            logger.error(f"GPT生成失败。返回码: {process.returncode}")
            logger.error(f"Stdout: {process.stdout}")
            logger.error(f"Stderr: {process.stderr}")
            raise Exception(f"GPT生成失败: {input_file}")
        else:
            logger.info(f"GPT生成脚本执行成功")

        # 检查并复制生成的文件到目标位置
        # 脚本生成的文件位置（使用unique_seed）
        default_output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
        source_valid_file = os.path.join(default_output_dir, f"crossovered0_frags_new_{unique_seed}.smi")
        source_complete_file = os.path.join(default_output_dir, f"crossovered0_fragsCom_new_{unique_seed}.smi")
        
        # 目标文件位置（使用更独特的文件名）
        target_valid_file = os.path.join(gpt_output_base_dir, f"{output_prefix}_gpt_valid_mols_gen{gen_num}.smi")
        target_complete_file = os.path.join(gpt_output_base_dir, f"{output_prefix}_gpt_complete_mols_gen{gen_num}.smi")
        
        # 复制文件
        import shutil
        try:
            if os.path.exists(source_valid_file):
                shutil.copy2(source_valid_file, target_valid_file)
                logger.info(f"已复制有效分子文件: {source_valid_file} -> {target_valid_file}")
            else:
                logger.warning(f"源文件不存在: {source_valid_file}")
                
            if os.path.exists(source_complete_file):
                shutil.copy2(source_complete_file, target_complete_file)
                logger.info(f"已复制完整生成文件: {source_complete_file} -> {target_complete_file}")
            else:
                logger.warning(f"源文件不存在: {source_complete_file}")
                
            # 清理源文件以避免下次冲突
            if os.path.exists(source_valid_file):
                os.remove(source_valid_file)
                logger.info(f"已清理源文件: {source_valid_file}")
            if os.path.exists(source_complete_file):
                os.remove(source_complete_file)
                logger.info(f"已清理源文件: {source_complete_file}")
                
        except Exception as e:
            logger.error(f"复制文件时出错: {str(e)}")
            raise Exception(f"GPT生成后文件处理失败: {str(e)}")

        # 检查目标文件是否存在
        if not os.path.exists(target_valid_file):
            logger.error(f"GPT生成后,目标文件未找到: {target_valid_file}")
            raise Exception(f"GPT生成后文件丢失: {target_valid_file}")

        logger.info(f"GPT生成成功。主要输出文件: {target_valid_file}")
        return target_valid_file
        
    finally:
        # 如果启用GPT串行模式，释放全局锁
        if gpt_serial_mode:
            _gpt_serial_lock.release()
            logger.info("已释放GPT串行锁")

def run_crossover(source_file, llm_file, output_file, gen_num, num_crossovers, logger):
    """运行分子交叉"""
    logger.info(f"开始分子交叉: 源文件 {source_file}, LLM生成文件 {llm_file}, 交叉生成新个体数目 {num_crossovers}")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        PYTHON_EXECUTABLE, crossover_script,
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
        PYTHON_EXECUTABLE, mutation_script,
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
        PYTHON_EXECUTABLE, filter_script,
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
        PYTHON_EXECUTABLE, docking_script,
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

def select_seeds_with_pareto_multi_objective(docking_output, seed_output, top_mols, diversity_mols, 
                                           logger, elitism_mols=1, prev_elite_mols=None):
    """
    使用NSGA-II帕累托算法进行多目标种子选择
    调用 operations/selecting/selecting_multi_demo.py 脚本实现
    
    Args:
        docking_output: 对接结果文件路径
        seed_output: 种子输出文件路径
        top_mols: 基于适应度选择的分子数量
        diversity_mols: 基于多样性选择的分子数量
        logger: 日志记录器
        elitism_mols: 精英分子数量
        prev_elite_mols: 上一代精英分子
    
    Returns:
        tuple: (种子文件路径, 新的精英分子字典)
    """
    logger.info(f"使用NSGA-II帕累托算法进行多目标种子选择")
    logger.info(f"选择 {top_mols} 个适应度种子和 {diversity_mols} 个多样性种子")
    
    # 读取对接结果以确定当前最优精英分子
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
        return None, None
    
    if not scores:
        logger.warning("对接结果中没有发现有效分数")
        return None, None
    
    # 确定当前代最优分子
    best_idx = np.argmin(scores)  # 对接分数越小越好
    current_best_mol = molecules[best_idx]
    current_best_score = scores[best_idx]
    
    # 处理精英分子保留逻辑
    if prev_elite_mols:
        prev_best_mol = list(prev_elite_mols.keys())[0]
        prev_best_score = list(prev_elite_mols.values())[0]
        
        # 比较当前代最好分子和上一代精英分子
        if current_best_score < prev_best_score:  # 对接分数越小越好
            new_elite_mols = {current_best_mol: current_best_score}
            logger.info(f"发现更好的分子，更新精英分子:")
            logger.info(f"上一代精英分子: {prev_best_mol} (得分: {prev_best_score:.4f})")
            logger.info(f"新的精英分子: {current_best_mol} (得分: {current_best_score:.4f})")
        else:
            new_elite_mols = {prev_best_mol: prev_best_score}
            logger.info(f"保留上一代精英分子:")
            logger.info(f"当前代最好分子: {current_best_mol} (得分: {current_best_score:.4f})")
            logger.info(f"保留的精英分子: {prev_best_mol} (得分: {prev_best_score:.4f})")
    else:
        new_elite_mols = {current_best_mol: current_best_score}
        logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score:.4f})")
    
    # 准备临时输出文件
    temp_pareto_output = seed_output.replace('.smi', '_pareto_temp.smi')
    
    # 调用多目标选择脚本
    selecting_script = os.path.join(PROJECT_ROOT, "operations/selecting/selecting_multi_demo.py")
    cmd = [
        PYTHON_EXECUTABLE, selecting_script,
        "--docked_file", docking_output,
        "--output_file", temp_pareto_output,
        "--n_select_fitness", str(top_mols),
        "--n_select_diversity", str(diversity_mols),
        "--verbose"
    ]
    
    logger.info(f"执行NSGA-II多目标选择命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"NSGA-II多目标选择失败: {process.stderr}")
        # 显示更详细的错误信息
        if process.stdout:
            logger.error(f"标准输出: {process.stdout}")
        if process.stderr:
            logger.error(f"标准错误: {process.stderr}")
        logger.error(f"回退到简单选择方法")
        # 回退到简单的基于对接分数的选择
        sorted_indices = np.argsort(scores)
        selected_mols = [molecules[i] for i in sorted_indices[:top_mols + diversity_mols]]
    else:
        logger.info(f"NSGA-II多目标选择成功")
        if process.stdout:
            logger.info(f"选择脚本输出:\n{process.stdout}")
        
        # 读取帕累托选择的结果
        selected_mols = []
        try:
            with open(temp_pareto_output, 'r') as f:
                for line in f:
                    mol = line.strip()
                    if mol:
                        selected_mols.append(mol)
        except Exception as e:
            logger.error(f"读取帕累托选择结果失败: {str(e)}")
            # 回退到简单选择
            sorted_indices = np.argsort(scores)
            selected_mols = [molecules[i] for i in sorted_indices[:top_mols + diversity_mols]]
    
    # 确保精英分子包含在种子中
    if new_elite_mols:
        elite_mol = list(new_elite_mols.keys())[0]
        if elite_mol not in selected_mols:
            selected_mols.insert(0, elite_mol)  # 将精英分子放在最前面
    
    # 去重但保持顺序
    unique_selected_mols = []
    seen = set()
    for mol in selected_mols:
        if mol not in seen:
            unique_selected_mols.append(mol)
            seen.add(mol)
    
    # 保存最终的种子分子
    with open(seed_output, 'w') as f:
        for mol in unique_selected_mols:
            f.write(f"{mol}\n")
    
    logger.info(f"NSGA-II帕累托种子选择完成:")
    logger.info(f"  - 精英分子: {len(new_elite_mols)}")
    logger.info(f"  - 帕累托选择分子: {len(unique_selected_mols)}")
    logger.info(f"  - 种子文件: {seed_output}")
    
    # 清理临时文件
    if os.path.exists(temp_pareto_output):
        os.remove(temp_pareto_output)
    
    return seed_output, new_elite_mols

def select_seeds_for_next_generation_simple(docking_output, seed_output, top_mols, diversity_mols, logger, elitism_mols=1, prev_elite_mols=None):
    """简化的基于对接分数的种子选择（作为备用方法）"""
    logger.info(f"使用简化的单目标种子选择方法")
    
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
        return None, None
    
    if not scores:
        logger.warning("对接结果中没有发现有效分数")
        return None, None
    
    # 按对接分数排序（对接分数越小越好）
    sorted_indices = np.argsort(scores)
    sorted_molecules = [molecules[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # 精英分子处理
    current_best_mol = sorted_molecules[0]
    current_best_score = sorted_scores[0]
    
    if prev_elite_mols:
        prev_best_mol = list(prev_elite_mols.keys())[0]
        prev_best_score = list(prev_elite_mols.values())[0]
        
        if current_best_score < prev_best_score:
            new_elite_mols = {current_best_mol: current_best_score}
            logger.info(f"更新精英分子: {current_best_mol} (得分: {current_best_score:.4f})")
        else:
            new_elite_mols = {prev_best_mol: prev_best_score}
            logger.info(f"保留精英分子: {prev_best_mol} (得分: {prev_best_score:.4f})")
    else:
        new_elite_mols = {current_best_mol: current_best_score}
        logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score:.4f})")
    
    # 选择适应度种子和多样性种子
    remaining_molecules = [mol for mol in sorted_molecules if mol not in new_elite_mols]
    fitness_seeds = remaining_molecules[:top_mols]
    
    # 简单的多样性选择
    diversity_seeds = []
    if diversity_mols > 0 and len(remaining_molecules) > top_mols:
        diversity_candidates = remaining_molecules[top_mols:]
        # 随机选择多样性种子
        diversity_count = min(diversity_mols, len(diversity_candidates))
        diversity_seeds = np.random.choice(diversity_candidates, diversity_count, replace=False).tolist()
    
    # 合并所有种子
    all_seeds = list(new_elite_mols.keys()) + fitness_seeds + diversity_seeds
    
    # 保存种子分子
    with open(seed_output, 'w') as f:
        for mol in all_seeds:
            f.write(f"{mol}\n")
    
    logger.info(f"简化种子选择完成，共选择 {len(all_seeds)} 个分子")
    return seed_output, new_elite_mols

def limit_population_size(input_file, max_size, output_file=None):
    """限制种群大小,保留前max_size个分子"""
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
        seed_output, new_elite_mols = select_seeds_with_pareto_multi_objective(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation, None
        )

        # 对第0代的对接结果进行评估
        evaluation_output_file = os.path.join(output_base, "generation_0_evaluation_metrics.txt")
        run_scoring_evaluation(docking_output, args.initial_population, evaluation_output_file, logger)

        return seed_output, new_elite_mols
    else:
        # 1. 读取上一代seed文件
        prev_seed_file = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_seeds.smi")
        logger.info(f"读取上一代种子文件: {prev_seed_file}")
        
        # 2. 分子分解
        decompose_prefix = f"target_{target}_gen{generation_num}_seed"
        decompose_output = run_decompose(prev_seed_file, decompose_prefix, logger, output_base)
        
        # 3. GPT生成新分子，并将这些新分子保留
        gpt_prefix = f"target_{target}_gen{generation_num}_seed"
        gpt_output = run_gpt_generation(decompose_output, gpt_prefix, generation_num, logger, output_base, args.gpt_serial_mode)
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
        seed_output, new_elite_mols = select_seeds_with_pareto_multi_objective(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation, prev_elite_mols
        )

        # 9. 对当前代的对接结果进行评估
        evaluation_output_file = os.path.join(output_base, f"generation_{generation_num}_evaluation_metrics.txt")
        run_scoring_evaluation(docking_output, args.initial_population, evaluation_output_file, logger)

        return seed_output, new_elite_mols

def run_scoring_evaluation(docked_file, initial_population_file, output_file, logger):
    """运行新种群的评估脚本."""
    logger.info(f"开始对种群进行评估: {docked_file}")
    scoring_script = os.path.join(PROJECT_ROOT, "operations/scoring/scoring_demo.py")
    cmd = [
        PYTHON_EXECUTABLE, scoring_script,
        "--current_population_docked_file", docked_file,
        "--initial_population_file", initial_population_file,
        "--output_file", output_file
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"种群评估失败: {process.stderr}")
        # Decide if this should raise an exception or just log an error
        # For now, just log and continue
    else:
        logger.info(f"种群评估完成，结果保存至: {output_file}")
        if process.stdout:
            logger.info(f"评估脚本输出:\n{process.stdout}")

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
                   target_args.multithread_mode,
                   center_x=target_args.center_x, center_y=target_args.center_y, center_z=target_args.center_z,
                   size_x=target_args.size_x, size_y=target_args.size_y, size_z=target_args.size_z)
        
        # 计算统计信息
        calculate_and_print_stats(docking_output, 0, logger)
        
        # 选择种子分子
        diversity_mols = max(0, target_args.diversity_mols_to_seed_first_generation)
        seed_output = os.path.join(gen0_output_dir, "generation_0_seeds.smi")
        seed_output, elite_mols = select_seeds_with_pareto_multi_objective(
            docking_output, seed_output, target_args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, target_args.elitism_mols_to_next_generation, None
        )
        
        # 对第0代的对接结果进行评估
        evaluation_output_file = os.path.join(gen0_output_dir, "generation_0_evaluation_metrics.txt")
        run_scoring_evaluation(docking_output, target_args.initial_population, evaluation_output_file, logger)
        
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
            decompose_prefix = f"target_{target}_gen{gen}_seed"
            decompose_output = run_decompose(prev_seed_file, decompose_prefix, logger, gen_output_dir)
            
            # 3. GPT生成新分子，并将这些新分子保留
            gpt_prefix = f"target_{target}_gen{gen}_seed"
            gpt_output = run_gpt_generation(decompose_output, gpt_prefix, gen, logger, gen_output_dir, target_args.gpt_serial_mode)
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
                       target_args.multithread_mode,
                       center_x=target_args.center_x, center_y=target_args.center_y, center_z=target_args.center_z,
                       size_x=target_args.size_x, size_y=target_args.size_y, size_z=target_args.size_z)
            calculate_and_print_stats(docking_output, gen, logger)
            
            # 8. 选择下一代种子
            diversity_mols = max(0, target_args.diversity_mols_to_seed_first_generation - 
                               (gen * target_args.diversity_seed_depreciation_per_gen))
            seed_output = os.path.join(gen_output_dir, f"generation_{gen}_seeds.smi")
            seed_output, elite_mols = select_seeds_with_pareto_multi_objective(
                docking_output, seed_output, target_args.top_mols_to_seed_next_generation, 
                diversity_mols, logger, target_args.elitism_mols_to_next_generation, elite_mols
            )
            
            # 9. 对当前代的对接结果进行评估
            evaluation_output_file = os.path.join(gen_output_dir, f"generation_{gen}_evaluation_metrics.txt")
            run_scoring_evaluation(docking_output, target_args.initial_population, evaluation_output_file, logger)
            
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
    if not _psutil_available:
        print("psutil库不可用,将使用os.cpu_count()返回所有核心数。")
        return os.cpu_count()
    try:
        # 获取CPU使用率小于80%的核心数量
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        available_cores = sum(1 for percent in cpu_percent if percent < 80)
        
        # 确保至少使用一个核心
        return max(1, available_cores)
    except Exception as e:
        # 如果无法获取CPU使用情况，默认使用全部核心
        print(f"使用psutil获取CPU使用情况时出错: {str(e)}，将使用全部核心")
        return os.cpu_count()

def run_docking(input_file, output_file, receptor_file, mgltools_path, logger, 
                num_processors=1, multithread_mode="serial",
                center_x=None, center_y=None, center_z=None, 
                size_x=None, size_y=None, size_z=None):
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
        PYTHON_EXECUTABLE, docking_script,
        "-i", input_file,
        "-r", receptor_file,
        "-o", output_file,
        "-m", mgltools_path,
        "--number_of_processors", str(num_processors),
        "--multithread_mode", multithread_mode
    ]
    
    # 添加对接盒子参数到命令中（如果已提供）
    if center_x is not None: cmd.extend(["--center_x", str(center_x)])
    if center_y is not None: cmd.extend(["--center_y", str(center_y)])
    if center_z is not None: cmd.extend(["--center_z", str(center_z)])
    if size_x is not None: cmd.extend(["--size_x", str(size_x)])
    if size_y is not None: cmd.extend(["--size_y", str(size_y)])
    if size_z is not None: cmd.extend(["--size_z", str(size_z)])

    logger.info(f"执行对接命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子对接失败: {process.stderr}")
        raise Exception("分子对接失败")
    
    logger.info(f"分子对接完成，生成文件: {output_file}")
    return output_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm_rga - 基于NSGA-II帕累托多目标优化的分子进化与生成流程')
    
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
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=5,
                       help='每代基于适应度选择进入下一代的分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=5,
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
    parser.add_argument('--gpt_serial_mode', action='store_true', default=False,
                        help='GPT生成是否使用串行模式。如果GPU显存不足,建议启用此选项。启用后,即使其他操作并行执行,GPT生成也会串行进行。')
    
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
    
    # 输出优化方法信息
    print("========== NSGA-II帕累托多目标优化配置 ==========")
    print("种子选择方法: NSGA-II帕累托算法")
    print("优化目标:")
    print("  1. 对接分数: 最小化（越小越好）")
    print("  2. QED分数: 最大化（药物相似性）") 
    print("  3. SA分数: 最小化（合成难度越小越好）")
    print("选择策略: 帕累托前沿 + 多策略选择")
    print("=" * 50)
    
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
    
    print("所有受体的NSGA-II帕累托多目标优化进化过程已完成！")

if __name__ == "__main__":
    main()
