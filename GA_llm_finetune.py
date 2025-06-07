#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_llm_finetune.py - 基于NSGA-II帕累托多目标优化的分子进化与生成流程 (改进版)

=============================================================================
程序功能：
=============================================================================
本程序是GA_llm分子优化框架的改进版本,采用NSGA-II帕累托算法进行多目标优化。
主要特性包括：

1. 多目标优化:同时优化对接分数、QED分数和SA分数
   - 对接分数：最小化（结合亲和力）
   - QED分数:最大化（药物相似性）  
   - SA分数:最小化（合成难度）

2. 遗传算法操作：
   - 分子分解
   - GPT辅助生成
   - 交叉操作
   - 变异操作
   - 精英保留

3. NSGA-II帕累托选择:真正的多目标优化，找到帕累托最优解集

=============================================================================
与原版本的主要改进：
=============================================================================
- 删除了基于对接分数单一指标的种子选择机制
- 引入真正的NSGA-II帕累托多目标优化
- 调用专门的多目标选择脚本 operations/selecting/selecting_multi_demo.py
- 保留精英分子机制但基于多目标评估

=============================================================================
使用示例：
=============================================================================

# 基本使用
python GA_llm_finetune.py --generations 5

# 自定义种子选择参数
python GA_llm_finetune.py \\
    --top_mols_to_seed_next_generation 15 \\
    --diversity_mols_to_seed_first_generation 15 \\
    --generations 10

# 使用并行处理
python GA_llm_finetune.py --number_of_processors 8 --multithread_mode multithreading

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
    log_file = os.path.join(output_dir, f"ga_evolution_{generation_num}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GA_llm_new")
def run_decompose(input_file, output_prefix, logger, current_gen_output_dir):
    """运行分子分解模块"""
    logger.info(f"开始分子分解: {input_file} (输出到: {current_gen_output_dir})")
    
    # 准备输出目录
    decompose_dir = os.path.join(current_gen_output_dir, "decompose_results")
    os.makedirs(decompose_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(decompose_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_dir, f"truncated_frags_{output_prefix}.smi")
    output_file4 = os.path.join(decompose_dir, f"decomposable_mols_{output_prefix}.smi")
    
    # 构建命令并执行
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
    
    if process.returncode != 0:
        logger.error(f"分子分解失败: {process.stderr}")
        raise Exception("分子分解失败")
    
    logger.info(f"分子分解完成，生成文件: {output_file3}")
    return output_file3

def run_gpt_generation(input_file, output_prefix, gen_num, logger, current_gen_output_dir):
    """运行GPT生成新分子"""
    logger.info(f"开始GPT生成: {input_file} (输出到: {current_gen_output_dir})")
    
    # 为GPT生成创建一个专用的子目录
    gpt_output_base_dir = os.path.join(current_gen_output_dir, "fragment_GPT_output")
    os.makedirs(gpt_output_base_dir, exist_ok=True)
    
    # GPT脚本的默认输出位置
    default_gpt_script_output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
    os.makedirs(default_gpt_script_output_dir, exist_ok=True) # 确保原始脚本的输出目录也存在

    # 构建命令并执行
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    
    # 使用一个更独特的seed，以避免在并行执行或多次运行时潜在的文件名冲突
    import time
    import hashlib
    timestamp_suffix = int(time.time() * 1000) % 10000
    prefix_hash = abs(hash(output_prefix)) % 1000
    unique_seed = int(f"{gen_num}{timestamp_suffix}{prefix_hash}")
    
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--device", "0",  # 注意：这里的设备ID可能需要根据实际情况调整或参数化
        "--seed", str(unique_seed) # 使用 unique_seed
    ]
    
    logger.info(f"执行GPT生成命令: {' '.join(cmd)} (使用 unique_seed: {unique_seed})")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"GPT生成失败: {process.stderr}")
        raise Exception("GPT生成失败")
    
    # generate_all.py 脚本输出的文件名模式
    # 例如: crossovered0_frags_new_{unique_seed}.smi
    # 我们关心的是 _new_ 部分的文件
    
    # 查找由 generate_all.py 生成的实际文件
    # 脚本通常在 PROJECT_ROOT/fragment_GPT/output/ 中生成文件
    # 文件名模式为 crossovered0_frags_new_{seed}.smi
    
    source_generated_file_pattern = os.path.join(default_gpt_script_output_dir, f"*_new_{unique_seed}.smi")
    generated_files_matched = glob.glob(source_generated_file_pattern)

    if not generated_files_matched:
        # 如果特定seed的文件找不到，尝试更通用的模式（尽管这不太应该发生，如果generate_all.py行为一致）
        logger.warning(f"使用unique_seed {unique_seed} 未找到预期的GPT输出文件。尝试查找 gen_num {gen_num} 的其他文件。")
        source_generated_file_pattern = os.path.join(default_gpt_script_output_dir, f"*_new_{gen_num}.smi")
        generated_files_matched = glob.glob(source_generated_file_pattern)

    if not generated_files_matched:
        logger.error(f"找不到任何GPT生成的输出文件。脚本目录: {default_gpt_script_output_dir}, 模式: {source_generated_file_pattern}")
        logger.error(f"请检查 fragment_GPT/generate_all.py 脚本的输出行为和位置。")
        raise Exception(f"找不到任何GPT生成的输出文件，生成可能失败 (seed: {unique_seed}, gen_num: {gen_num})")

    # 如果有多个匹配（理论上不应该，如果seed是唯一的），选择最新的一个
    source_output_file = max(generated_files_matched, key=os.path.getmtime)
    logger.info(f"找到GPT脚本生成的原始文件: {source_output_file}")

    # 定义目标文件路径（在当前代的 fragment_GPT_output 目录下）
    # 使用 output_prefix 保证文件名在代内唯一性，例如区分不同target的并行运行时
    target_output_filename = f"{output_prefix}_gpt_generated_gen{gen_num}.smi"
    final_output_file = os.path.join(gpt_output_base_dir, target_output_filename)
    
    # 复制并重命名文件到目标位置
    import shutil
    try:
        shutil.copy2(source_output_file, final_output_file)
        logger.info(f"已将GPT生成文件从 {source_output_file} 复制到 {final_output_file}")
        
        # （可选）清理源文件，以避免积累和潜在的下次冲突
        # os.remove(source_output_file)
        # logger.info(f"已清理源文件: {source_output_file}")

    except Exception as e:
        logger.error(f"复制GPT生成文件时出错: {str(e)}")
        raise Exception(f"GPT生成后文件处理失败: {str(e)}")

    logger.info(f"GPT生成完成,输出文件: {final_output_file}")
    return final_output_file

def run_crossover(source_file, llm_file, output_file, gen_num, num_crossovers, logger):
    """运行分子交叉"""
    logger.info(f"开始分子交叉: 源文件 {source_file}, LLM生成文件 {llm_file}, 交叉生成新个体数目 {num_crossovers}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        "python", crossover_script,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_attempts", str(num_crossovers)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子交叉失败: {process.stderr}")
        raise Exception("分子交叉失败")
    
    logger.info(f"分子交叉完成，生成文件: {output_file}")
    return output_file

def run_mutation(input_file, llm_file, output_file, num_mutations, logger):
    """运行分子变异"""
    logger.info(f"开始分子变异: 输入文件 {input_file}, LLM生成文件 {llm_file}, 变异生成新个体数目 {num_mutations}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo_finetune.py")
    cmd = [
        "python", mutation_script,
        "--input_file", input_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--num_mutations", str(num_mutations)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子变异失败: {process.stderr}")
        raise Exception("分子变异失败")
    
    logger.info(f"分子变异完成，生成文件: {output_file}")
    return output_file

def run_filter(input_file, output_file, logger, args):
    """运行分子过滤"""
    logger.info(f"开始分子过滤: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建过滤器参数列表
    filter_params = []
    
    # 检查每个过滤器参数并添加到命令行
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
    
    # 添加自定义过滤器
    if args.alternative_filter:
        for filter_entry in args.alternative_filter:
            filter_params.extend(["--alternative_filter", filter_entry])
    
    # 如果没有指定任何过滤器，记录一条警告
    if not filter_params and not args.No_Filters:
        logger.warning("没有指定任何过滤器参数，将使用默认过滤器")
    
    # 构建命令并执行
    filter_script = os.path.join(PROJECT_ROOT, "operations/filter/filter_demo.py")
    cmd = [
        "python", filter_script,
        "--input", input_file,
        "--output", output_file
    ]
    
    # 添加过滤器参数
    cmd.extend(filter_params)
    
    logger.info(f"执行过滤命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子过滤失败: {process.stderr}")
        raise Exception("分子过滤失败")
    
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
        logger.info(f"已将综合得分文件复制到: {output_file}")
    else:
        logger.error(f"综合得分文件不存在: {combined_scores_file}")
        raise Exception("多受体对接失败")
    
    logger.info(f"多受体对接流程完成，结果保存至: {output_file}")
    return output_file

def dock_molecule(mol_idx, mol_smiles, args, temp_dir, logger):
    """对单个分子进行对接"""
    try:
        # 创建临时文件
        temp_input = os.path.join(temp_dir, f"temp_input_{mol_idx}.smi")
        temp_output = os.path.join(temp_dir, f"temp_output_{mol_idx}.smi")
        
        # 写入分子到临时文件
        with open(temp_input, 'w') as f:
            f.write(mol_smiles)
        
        # 构建对接命令
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "-i", temp_input,
            "-r", args.receptor_file,
            "-o", temp_output,
            "-m", args.mgltools_path,
            "--max_failures", "5"
        ]
        
        # 执行对接
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.warning(f"分子 {mol_idx} 对接失败: {process.stderr}")
            return None
        
        # 读取对接结果
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                result = f.read().strip()
            if result:
                return result
        
        return None
        
    except Exception as e:
        logger.error(f"分子 {mol_idx} 对接过程出错: {str(e)}")
        return None
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except:
            pass

def run_docking(input_file, output_file, receptor_file, mgltools_path, logger, num_processors=1, multithread_mode="serial"):
    """运行分子对接，支持并行处理"""
    logger.info(f"开始分子对接: {input_file}, 处理器数量: {num_processors}, 模式: {multithread_mode}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定处理器数量 - 如果为-1或大于可用CPU数量，则使用所有可用CPU
    available_cpus = multiprocessing.cpu_count()
    if num_processors == -1 or num_processors > available_cpus:
        num_processors = available_cpus
        logger.info(f"自动设置使用所有可用的CPU核心: {num_processors}")
    
    # 根据处理器数量自动选择并行模式
    if num_processors > 1 and multithread_mode == "serial":
        logger.info(f"检测到使用多核({num_processors})但模式为serial,自动切换为multithreading模式")
        multithread_mode = "multithreading"
        
    # 如果选择串行模式或只使用一个处理器，使用原始的对接方法
    if multithread_mode == "serial" or num_processors == 1:
        logger.info("使用串行模式进行对接")
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "-i", input_file,
            "-r", receptor_file,
            "-o", output_file,
            "-m", mgltools_path,
            "--max_failures", "5"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"分子对接失败: {process.stderr}")
            raise Exception("分子对接失败")
        
        logger.info(f"分子对接完成，生成文件: {output_file}")
        return output_file
    
    # 并行处理
    logger.info(f"使用并行模式进行对接，处理器数量: {num_processors}")
    
    # 读取输入文件中的分子
    with open(input_file, 'r') as f:
        molecules = [line for line in f.readlines() if line.strip()]
    
    total_molecules = len(molecules)
    logger.info(f"共有 {total_molecules} 个分子需要对接")
    
    # 创建临时目录存放分割后的文件
    temp_dir = os.path.join(output_dir, "temp_docking")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 设置工作函数参数
    dock_func = partial(dock_molecule, args=argparse.Namespace(
        receptor_file=receptor_file,
        mgltools_path=mgltools_path
    ), temp_dir=temp_dir, logger=logger)
    
    # 计算每个处理器应该处理的分子数量，确保负载平衡
    molecules_per_processor = max(1, total_molecules // num_processors)
    
    # 并行执行对接
    results = []
    start_time = time.time()
    
    # 优化：根据分子数量和处理器数量自动调整最优的批处理大小
    batch_size = max(1, min(100, molecules_per_processor))
    
    # 分子任务分组，优化负载均衡
    molecule_batches = []
    for i in range(0, total_molecules, batch_size):
        end = min(i + batch_size, total_molecules)
        molecule_batches.append((i, molecules[i:end]))
    
    logger.info(f"将 {total_molecules} 个分子分为 {len(molecule_batches)} 批进行处理，每批大约 {batch_size} 个分子")
    
    # 优化：使用批处理方式进行对接
    if multithread_mode == "multithreading":
        logger.info(f"使用多线程模式，线程数: {num_processors}")
        with ThreadPoolExecutor(max_workers=num_processors) as executor:
            # 批量提交任务，改善负载均衡
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx
            
            # 处理结果时显示进度
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1
                
                # 定期更新进度信息
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    else:  # 多进程模式
        logger.info(f"使用多进程模式，进程数: {num_processors}")
        # 使用spawn上下文避免潜在的内存泄漏问题
        mp_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_processors, mp_context=mp_context) as executor:
            # 批量提交任务
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx
            
            # 处理结果时显示进度
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1
                
                # 定期更新进度信息
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"对接计算完成，总耗时: {total_time:.2f}秒，"
               f"平均每个分子: {total_time/total_molecules:.2f}秒，"
               f"总成功率: {len(results)/total_molecules*100:.1f}%")
    
    # 合并结果到输出文件
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    logger.info(f"并行对接完成，成功对接 {len(results)}/{total_molecules} 个分子，结果保存至: {output_file}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_file

def run_analysis(input_file, output_prefix, gen_num, logger):
    """运行对接结果分析"""
    logger.info(f"开始对接结果分析: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(input_file)
    
    # 构建命令并执行
    analysis_script = os.path.join(PROJECT_ROOT, "operations/docking/analyse_result_0.py")
    cmd = [
        "python", analysis_script,
        "--input", input_file,
        "--output", output_dir,
        "--prefix", f"generation_{gen_num}"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"对接结果分析失败: {process.stderr}")
        raise Exception("对接结果分析失败")
    
    logger.info(f"对接结果分析完成，结果保存至: {output_dir}/generation_{gen_num}_stats.txt")
    return f"{output_dir}/generation_{gen_num}_sorted.smi"

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
        "python", selecting_script,
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
    logger.info(f"使用简化的单目标种子选择方法（备用）")
    
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

# 保留原有函数的包装器以向后兼容
def select_seeds_for_next_generation(docking_output, seed_output, top_mols, diversity_mols, logger, elitism_mols=1, prev_elite_mols=None):
    """
    种子选择函数的包装器，默认使用NSGA-II帕累托多目标选择
    如果失败则回退到简单选择
    """
    try:
        return select_seeds_with_pareto_multi_objective(
            docking_output, seed_output, top_mols, diversity_mols, 
            logger, elitism_mols, prev_elite_mols
        )
    except Exception as e:
        logger.error(f"帕累托多目标选择失败: {str(e)}")
        logger.warning("回退到简化的单目标选择方法")
        return select_seeds_for_next_generation_simple(
            docking_output, seed_output, top_mols, diversity_mols, 
            logger, elitism_mols, prev_elite_mols
        )

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

def run_scoring_evaluation(docked_file, initial_population_file, output_file, logger):
    """运行新种群的评估脚本."""
    logger.info(f"开始对种群进行评估: {docked_file}")
    scoring_script = os.path.join(PROJECT_ROOT, "operations/scoring/scoring_demo.py")
    cmd = [
        "python", scoring_script,
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

def run_evolution(generation_num, args, logger, prev_elite_mols=None):
    """执行一次完整的进化迭代，支持精英保留机制"""
    logger.info(f"开始第 {generation_num} 代进化")
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)

    # 0. 确定当前代的种群文件
    if generation_num == 0:
        current_population = args.initial_population
        # 初代直接多受体对接+scoring
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_multi_receptor_docking_pipeline(current_population, docking_output, args.targets, logger)
        calculate_and_print_stats(docking_output, generation_num, logger)
        # 选seed
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation
        )
        
        # 在选完seed后，对当前代的对接结果进行评估
        evaluation_output_file = os.path.join(output_base, f"generation_{generation_num}_evaluation_metrics.txt")
        run_scoring_evaluation(docking_output, args.initial_population, evaluation_output_file, logger)

        return seed_output, new_elite_mols
    else:
        # 1. 读取上一代seed，但排除精英分子
        prev_seed_file = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_seeds.smi")
        non_elite_molecules = []
        with open(prev_seed_file, 'r') as f:
            for line in f:
                mol = line.strip()
                if mol and (prev_elite_mols is None or mol not in prev_elite_mols):
                    non_elite_molecules.append(mol)
        
        # 2. 只对非精英分子进行decompose+gpt生成
        temp_seed_file = os.path.join(output_base, "temp_non_elite_seeds.smi")
        with open(temp_seed_file, 'w') as f:
            for mol in non_elite_molecules:
                f.write(f"{mol}\n")
        
        decompose_output = run_decompose(temp_seed_file, f"gen{generation_num}_seed", logger, output_base)
        gpt_output = run_gpt_generation(decompose_output, f"gen{generation_num}_seed", generation_num, logger, output_base)
        
        # 3. 交叉（只使用非精英分子）
        crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
        run_crossover(temp_seed_file, gpt_output, crossover_output, generation_num, args.num_crossovers, logger)
        
        # 4. 变异（只使用非精英分子）
        mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
        run_mutation(temp_seed_file, gpt_output, mutation_output, args.num_mutations, logger)
        
        # 5. 合并新种群（精英分子 + 新生成的分子）
        new_population_file = os.path.join(output_base, f"generation_{generation_num}_new_population.smi")
        with open(new_population_file, 'w') as fout:
            # 首先写入精英分子（如果有的话）
            if prev_elite_mols:
                for mol, score in prev_elite_mols.items():
                    fout.write(f"{mol}\n")
                logger.info(f"已将上一代精英分子 {list(prev_elite_mols.keys())[0]} (得分: {list(prev_elite_mols.values())[0]}) 加入新种群")
            
            # 然后写入交叉和变异产生的新分子
            for fname in [crossover_output, mutation_output]:
                with open(fname, 'r') as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
        
        # 6. docking+scoring
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_multi_receptor_docking_pipeline(new_population_file, docking_output, args.targets, logger)
        calculate_and_print_stats(docking_output, generation_num, logger)
        
        # 7. 选seed
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation, prev_elite_mols
        )
        
        # 在选完seed后，对当前代的对接结果进行评估
        evaluation_output_file = os.path.join(output_base, f"generation_{generation_num}_evaluation_metrics.txt")
        # 使用 args.initial_population 作为新颖性计算的基准
        run_scoring_evaluation(docking_output, args.initial_population, evaluation_output_file, logger)

        # 清理临时文件
        if os.path.exists(temp_seed_file):
            os.remove(temp_seed_file)
            
        return seed_output, new_elite_mols

def run_evolution_for_target(target, args, generations):
    """为单个受体运行完整的进化过程"""
    # 为当前受体创建单独的输出目录
    target_output_dir = os.path.join(args.output_dir, f"target_{target}")
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 创建当前受体的参数副本，并修改输出目录和目标受体
    target_args = argparse.Namespace(**vars(args))
    target_args.output_dir = target_output_dir
    target_args.targets = [target]  # 只处理当前受体
    
    print(f"======== 开始针对受体 {target} 的进化过程 ========")
    
    # 执行多代进化
    logger = setup_logging(target_output_dir, 0)
    elite_mols = None
    
    try:
        logger.info(f"开始第0代进化 (对初始种群直接进行对接 - 目标受体: {target})")
        start_time = time.time()
        
        final_output, elite_mols = run_evolution(0, target_args, logger)
        
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
            
            final_output, elite_mols = run_evolution(gen, target_args, logger, elite_mols)
            
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {final_output}")
            
        except Exception as e:
            logger.error(f"第 {gen} 代进化失败: {str(e)}")
            print(f"受体 {target} 的第 {gen} 代进化失败: {str(e)}")
            break  # 如果某一代失败，跳过此受体的后续代
    
    print(f"======== 受体 {target} 的进化过程完成 ========")
    return target

def get_available_cpu_count():
    """获取当前系统可用的CPU核心数量"""
    try:
        import psutil
        # 获取CPU使用率小于80%的核心数量
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        available_cores = sum(1 for percent in cpu_percent if percent < 80)
        # 确保至少使用一个核心
        return max(1, available_cores)
    except ImportError:
        print("psutil库不可用,将使用os.cpu_count()返回所有核心数。")
        return os.cpu_count()
    except Exception as e:
        # 如果无法获取CPU使用情况，默认使用全部核心
        print(f"使用psutil获取CPU使用情况时出错: {str(e)}，将使用全部核心")
        return os.cpu_count()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm_finetune - 基于NSGA-II帕累托多目标优化的分子进化与生成流程 (改进版)')
    
    # 输出优化方法信息
    print("========== NSGA-II帕累托多目标优化配置 (改进版) ==========")
    print("种子选择方法: NSGA-II帕累托算法")
    print("优化目标:")
    print("  1. 对接分数: 最小化（越小越好）")
    print("  2. QED分数: 最大化（药物相似性）") 
    print("  3. SA分数: 最小化（合成难度越小越好）")
    print("选择策略: 帕累托前沿 + 多策略选择")
    print("与原版本区别: 从单目标DS优化改为真正的多目标帕累托优化")
    print("支持多受体: 同时对10种受体蛋白进行对接优化")
    print("目录结构: 每个受体创建独立的target_*目录")
    print("=" * 60)
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=5, 
                        help='进化代数(不包括第0代,总共会生成6代:generation_0到generation_5)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'output_finetune'),
                        help='基础输出目录,每个受体会在此目录下创建target_*子目录')
    parser.add_argument('--initial_population', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'datasets/source_compounds/naphthalene_smiles.smi'),
                        help='初始种群文件路径')
    
    # 对接参数 - 多受体支持
    parser.add_argument('--targets', nargs='+', 
                        default=['4r6e', '3pbl', '1iep', '2rgp', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11'], 
                        help='受体蛋白列表，每个受体将创建独立的进化流程')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='是否并行处理不同受体的进化过程')
    parser.add_argument('--max_workers', type=int, default=-1,
                        help='并行处理时的最大进程数，默认为-1表示自动检测并使用所有空闲CPU核心')
    parser.add_argument('--mgltools_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'mgltools_x86_64Linux2_1.5.6'),
                        help='MGLTools安装路径')
    
    # 进化参数
    parser.add_argument('--num_crossovers', type=int, default=50)
    parser.add_argument('--num_mutations', type=int, default=50)
    parser.add_argument('--number_of_crossovers_first_generation', type=int,
                       help='第0代中通过交叉产生的配体数量,如果未指定则默认使用num_crossovers的值')
    parser.add_argument('--number_of_mutants_first_generation', type=int,
                       help='第0代中通过变异产生的配体数量,如果未指定则默认使用num_mutations的值')
    parser.add_argument('--max_population', type=int, default=0,
                       help='控制每代种群的最大数量,设置为0表示不限制(可能导致种群规模迅速增长）')
    
    # 种子选择参数
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=50,
                       help='每代基于适应度选择进入下一代的分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=50,
                       help='第0代基于多样性选择进入下一代的分子数量')
    parser.add_argument('--diversity_seed_depreciation_per_gen', type=int, default=10,
                       help='每代多样性种子数量的递减值')
    parser.add_argument('--elitism_mols_to_next_generation', type=int, default=1,
                       help='每代保留的精英分子数量，这些分子将直接进入下一代而不进行进化操作')
    
    # 并行处理参数
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于并行计算的处理器数量。设置为-1表示自动检测并使用所有可用CPU核心(推荐）。')
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
    
    # 如果number_of_processors为-1，不在此处设置具体值，而是在run_docking函数中动态设置
    if args.number_of_processors == -1:
        print(f"将使用动态检测的CPU数量,在每次对接时自动设置")
    else:
        available_cpus = multiprocessing.cpu_count()
        if args.number_of_processors > available_cpus:
            print(f"指定的处理器数量({args.number_of_processors})超过系统可用CPU数量({available_cpus})，将使用所有可用CPU")
            args.number_of_processors = available_cpus
        else:
            print(f"将使用指定的{args.number_of_processors}个CPU进行计算")
    
    # 如果使用多核但未指定多线程模式，自动切换为multithreading模式
    if args.number_of_processors != 1 and args.multithread_mode == "serial":
        print(f"检测到可能使用多核但模式为serial,自动切换为multithreading模式")
        args.multithread_mode = "multithreading"
    
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
        max_workers = multiprocessing.cpu_count()
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

"""
=============================================================================
详细使用说明：
=============================================================================

1. 基本运行（为所有10种受体创建独立的target_*目录）：
   python GA_llm_finetune.py

2. 指定特定的受体蛋白子集：
   python GA_llm_finetune.py --targets 4r6e 3pbl 1iep

3. 并行处理所有受体：
   python GA_llm_finetune.py --parallel --max_workers 4

4. 指定进化代数：
   python GA_llm_finetune.py --generations 10

5. 自定义种子选择参数：
   python GA_llm_finetune.py \\
       --top_mols_to_seed_next_generation 20 \\
       --diversity_mols_to_seed_first_generation 15 \\
       --diversity_seed_depreciation_per_gen 1

6. 使用并行处理加速：
   python GA_llm_finetune.py \\
       --number_of_processors 8 \\
       --multithread_mode multithreading

7. 自定义输出目录和初始种群：
   python GA_llm_finetune.py \\
       --output_dir ./my_output \\
       --initial_population ./my_compounds.smi

8. 限制种群大小防止过度增长：
   python GA_llm_finetune.py \\
       --max_population 1000

9. 组合使用（推荐）：
   python GA_llm_finetune.py \\
       --targets 4r6e 3pbl 1iep 2rgp \\
       --parallel --max_workers 4 \\
       --generations 8 \\
       --top_mols_to_seed_next_generation 15

=============================================================================
输出文件说明：
=============================================================================

目录结构（每个受体独立）：
output_finetune/
├── target_4r6e/
│   ├── generation_0/
│   │   ├── generation_0_docked.smi
│   │   ├── generation_0_seeds.smi
│   │   └── generation_0_evaluation_metrics.txt
│   ├── generation_1/
│   │   ├── generation_1_docked.smi
│   │   ├── generation_1_seeds.smi
│   │   └── generation_1_evaluation_metrics.txt
│   └── ...
├── target_3pbl/
│   └── ...
└── target_1iep/
    └── ...

每代进化会在各自受体目录下创建以下文件：
- generation_X_docked.smi: 多受体综合对接结果（SMILES + 综合对接分数）
- generation_X_seeds.smi: 帕累托选择的种子分子
- generation_X_evaluation_metrics.txt: 种群评估指标
- multi_receptor_docking/: 包含各个受体的详细对接结果

关键改进：
- 独立目录结构：每个受体有独立的target_*目录，避免结果混淆
- 并行处理支持：可以并行处理多个受体的进化过程
- NSGA-II帕累托选择：同时优化对接分数、QED和SA分数
- 自动回退机制：如果帕累托选择失败，会使用简单的对接分数排序
- 保留精英分子机制确保最优解不会丢失

=============================================================================
"""
