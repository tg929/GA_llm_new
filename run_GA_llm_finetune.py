#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import argparse
import glob
import re
import shutil
from pathlib import Path
import numpy as np

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def parse_docking_score(file_path):
    """解析对接结果文件，获取最佳得分"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                # 文件格式通常为 "SMILES score"，取第一行的得分
                parts = lines[0].strip().split()
                if len(parts) >= 2:
                    return float(parts[1]), parts[0]  # 返回得分和SMILES
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {str(e)}")
    return None, None

def find_best_molecule(output_dir):
    """在所有代中找到得分最好的分子"""
    best_score = float('inf')  # 对接得分越低越好
    best_smiles = None
    best_gen = -1
    
    # 检查每一代的对接结果
    for gen in range(6):  # 0-5代
        sorted_file = os.path.join(output_dir, f"generation_{gen}", f"generation_{gen}_sorted.smi")
        if os.path.exists(sorted_file):
            score, smiles = parse_docking_score(sorted_file)
            if score is not None and score < best_score:
                best_score = score
                best_smiles = smiles
                best_gen = gen
    
    return best_score, best_smiles, best_gen

def run_single_experiment(exp_id, output_base_dir, args):
    """运行单次实验"""
    print(f"开始实验 #{exp_id}")
    start_time = time.time()
    
    # 为本次实验创建输出目录
    exp_output_dir = os.path.join(output_base_dir, f"exp_{exp_id}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 获取完整的绝对路径确保路径传递正确
    receptor_path = os.path.abspath(os.path.join(PROJECT_ROOT, "tutorial/PARP/4r6eA_PARP1_prepared.pdb"))
    mgltools_path = os.path.abspath(os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6"))
    initial_pop_path = os.path.abspath(os.path.join(PROJECT_ROOT, "datasets/source_compounds/naphthalene_smiles.smi"))
    
    # 验证路径是否存在
    if not os.path.exists(receptor_path):
        print(f"错误: 实验 #{exp_id} 受体文件不存在: {receptor_path}")
        return None, None, None
    
    if not os.path.exists(mgltools_path):
        print(f"错误: 实验 #{exp_id} MGLTools路径不存在: {mgltools_path}")
        return None, None, None
    
    if not os.path.exists(initial_pop_path):
        print(f"错误: 实验 #{exp_id} 初始种群文件不存在: {initial_pop_path}")
        return None, None, None
    
    # 构建命令
    cmd = [
        "python", "GA_llm_finetune.py",
        "--output_dir", exp_output_dir,
        "--generations", "5",  # 默认5代
        "--receptor_file", receptor_path,
        "--mgltools_path", mgltools_path,
        "--initial_population", initial_pop_path
    ]
    
    # 添加额外的参数
    if args.number_of_processors:
        cmd.extend(["--number_of_processors", str(args.number_of_processors)])
    
    # 添加过滤器参数 - 修正参数传递方式
    cmd.append("--No_Filters")  # 布尔标志参数无需值
    
    # 添加多线程模式参数，确保每个实验使用合适的模式
    cmd.extend(["--multithread_mode", "multithreading"])
    
    # 运行命令
    try:
        print(f"实验 #{exp_id} 执行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # 保存完整日志，便于调试
        with open(os.path.join(exp_output_dir, f"exp_{exp_id}_log.txt"), 'w') as f:
            f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
        
        if process.returncode != 0:
            print(f"实验 #{exp_id} 失败: {process.stderr}")
            return None, None, None
    except Exception as e:
        print(f"运行实验 #{exp_id} 时出错: {str(e)}")
        return None, None, None
    
    # 查找最佳结果
    best_score, best_smiles, best_gen = find_best_molecule(exp_output_dir)
    
    end_time = time.time()
    print(f"实验 #{exp_id} 完成，耗时: {end_time - start_time:.2f}秒，最佳得分: {best_score}，来自第{best_gen}代")
    
    return best_score, best_smiles, best_gen

def main():
    parser = argparse.ArgumentParser(description='批量运行GA_llm_finetune.py并保存最佳结果')
    parser.add_argument('--start_id', type=int, default=0, help='起始实验ID')
    parser.add_argument('--num_experiments', type=int, default=100, help='要运行的实验数量')
    parser.add_argument('--output_dir', type=str, default='batch_experiments', help='输出目录')
    parser.add_argument('--number_of_processors', type=int, help='每个实验使用的处理器数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建结果文件
    results_file = os.path.join(args.output_dir, "best_results.smi")
    
    # 运行多次实验
    for i in range(args.start_id, args.start_id + args.num_experiments):
        best_score, best_smiles, best_gen = run_single_experiment(i, args.output_dir, args)
        
        # 保存结果
        if best_score is not None and best_smiles is not None:
            with open(results_file, 'a') as f:
                f.write(f"{best_smiles}\t{best_score:.4f}\texp_{i}_gen_{best_gen}\n")
        
        # 每10次实验后保存一个备份
        if (i + 1) % 10 == 0:
            backup_file = os.path.join(args.output_dir, f"best_results_backup_{i+1}.smi")
            if os.path.exists(results_file):
                shutil.copy2(results_file, backup_file)
                print(f"已创建备份: {backup_file}")

if __name__ == "__main__":
    main()
