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
def run_decompose(input_file, output_prefix, logger):
    """运行分子分解模块"""
    logger.info(f"开始分子分解: {input_file}")
    
    # 准备输出目录
    decompose_dir = os.path.join(PROJECT_ROOT, "datasets/decompose/decompose_results")
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

def run_gpt_generation(input_file, output_prefix, gen_num, logger):
    """运行GPT生成新分子"""
    logger.info(f"开始GPT生成: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--device", "0",
        "--seed", str(gen_num)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"GPT生成失败: {process.stderr}")
        raise Exception("GPT生成失败")
    
    # 修改文件查找策略
    # 首先查找所有匹配的文件模式
    possible_files = [
        os.path.join(output_dir, f"crossovered{gen_num}_frags_new_{gen_num}.smi"),  # 期望的命名模式
        os.path.join(output_dir, f"crossovered{output_prefix}_frags_new_{gen_num}.smi"),  # 使用output_prefix
    ]
    
    # 添加所有后缀为_new_{gen_num}.smi的文件
    suffix_pattern_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                           if f.endswith(f"_new_{gen_num}.smi")]
    
    # 合并所有可能的文件并去重
    all_candidate_files = list(set(possible_files + suffix_pattern_files))
    
    # 检查这些文件是否存在
    existing_files = [f for f in all_candidate_files if os.path.exists(f)]
    
    if existing_files:
        # 如果有多个文件，选择最新的一个
        output_file = max(existing_files, key=lambda f: os.path.getmtime(f))
        logger.info(f"找到GPT生成的输出文件: {output_file}")
    else:
        # 没有找到任何匹配的文件，查找备用模式
        backup_files = []
        for pattern in [f"*_new_{gen_num}.smi", f"*_frags_new_*.smi"]:
            matching_files = glob.glob(os.path.join(output_dir, pattern))
            backup_files.extend(matching_files)
        
        if backup_files:
            output_file = max(backup_files, key=lambda f: os.path.getmtime(f))
            logger.info(f"通过备用模式找到GPT生成的输出文件: {output_file}")
        else:
            raise Exception(f"找不到任何GPT生成的输出文件，生成可能失败")
    
    logger.info(f"GPT生成完成，输出文件: {output_file}")
    return output_file

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
        # 初代直接docking+scoring
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_docking(current_population, docking_output, args.receptor_file, args.mgltools_path, logger, args.number_of_processors, args.multithread_mode)
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
        
        decompose_output = run_decompose(temp_seed_file, f"gen{generation_num}_seed", logger)
        gpt_output = run_gpt_generation(decompose_output, f"gen{generation_num}_seed", generation_num, logger)
        
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
        run_docking(new_population_file, docking_output, args.receptor_file, args.mgltools_path, logger, args.number_of_processors, args.multithread_mode)
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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm - 分子进化与生成流程')
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=5, 
                        help='进化代数(不包括第0代,总共会生成6代:generation_0到generation_5)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'output'),
                        help='输出目录')
    parser.add_argument('--initial_population', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'datasets/source_compounds/naphthalene_smiles.smi'),
                        help='初始种群文件路径')
    
    # 对接参数
    parser.add_argument('--receptor_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'tutorial/PARP/4r6eA_PARP1_prepared.pdb'),
                        help='受体PDB文件路径')
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
    
    # 创建输出目录
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
    
    # 执行多代进化
    # 先运行第0代（进行交叉和变异操作后再对接）
    logger = setup_logging(args.output_dir, 0)
    try:
        # 确定第0代使用的交叉变异生成新分子数目
        if args.number_of_crossovers_first_generation is None:
            args.number_of_crossovers_first_generation = args.num_crossovers
            logger.info(f"第0代交叉生成新个体数未指定,使用默认值: {args.num_crossovers}")
        
        if args.number_of_mutants_first_generation is None:
            args.number_of_mutants_first_generation = args.num_mutations
            logger.info(f"第0代变异生成新个体数未指定,使用默认值: {args.num_mutations}")
            
        logger.info(f"开始第0代(对初始种群进行交叉变异后对接)")
        logger.info(f"第0代将通过交叉生成 {args.number_of_crossovers_first_generation} 个新分子和 通过变异生成{args.number_of_mutants_first_generation} 个新分子")
        start_time = time.time()
        
        final_output, elite_mols = run_evolution(0, args, logger)
        
        end_time = time.time()
        logger.info(f"第0代完成,耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"第0代失败: {str(e)}")
        elite_mols = None
    
    # 执行后续5代进化
    for gen in range(1, args.generations + 1):
        logger = setup_logging(args.output_dir, gen)
        try:
            logger.info(f"开始第 {gen} 代进化")
            start_time = time.time()
            
            # 如果前一代种群存在且超过限制大小，先限制它
            if args.max_population > 0:
                prev_gen_file = os.path.join(args.output_dir, f"generation_{gen-1}", f"generation_{gen-1}_docked.smi")
                if os.path.exists(prev_gen_file):
                    with open(prev_gen_file, 'r') as f:
                        prev_count = sum(1 for line in f if line.strip())
                    if prev_count > args.max_population:
                        limit_population_size(prev_gen_file, args.max_population)
                        logger.info(f"第{gen-1}代种群已从{prev_count}限制为{args.max_population}")
            
            final_output, elite_mols = run_evolution(gen, args, logger, elite_mols)
            
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {final_output}")
            
        except Exception as e:
            logger.error(f"第 {gen} 代进化失败: {str(e)}")
            break

if __name__ == "__main__":
    main()
