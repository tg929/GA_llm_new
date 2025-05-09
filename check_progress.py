#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import time
import re
from pathlib import Path

def count_experiments_in_file(file_path):
    """计算结果文件中的实验数量"""
    try:
        with open(file_path, 'r') as f:
            return len(f.readlines())
    except:
        return 0

def check_running_processes():
    """检查正在运行的GA_llm_finetune.py进程数量"""
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        output = result.stdout
        ga_processes = [line for line in output.split('\n') if 'GA_llm_finetune.py' in line and 'grep' not in line]
        return len(ga_processes)
    except:
        return -1  # 表示无法确定

def parse_log_file(log_file):
    """解析日志文件获取进度信息"""
    if not os.path.exists(log_file):
        return 0, 0, None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # 查找最后一个完成的实验
        exp_pattern = r"实验 #(\d+) 完成"
        matches = re.findall(exp_pattern, content)
        last_exp = int(matches[-1]) if matches else -1
        
        # 查找最后一个开始的实验
        start_pattern = r"开始实验 #(\d+)"
        matches = re.findall(start_pattern, content)
        last_start = int(matches[-1]) if matches else -1
        
        # 查找最后更新时间
        last_modified = time.ctime(os.path.getmtime(log_file))
        
        return last_exp + 1, last_start + 1, last_modified
    except Exception as e:
        print(f"解析日志文件 {log_file} 出错: {str(e)}")
        return 0, 0, None

def check_progress(base_dir="output_3000experiments"):
    """检查所有批次的进度"""
    print("检查实验进度...\n")
    
    # 查找所有批次目录
    batch_dirs = sorted(glob.glob(os.path.join(base_dir, "batch_*")))
    
    if not batch_dirs:
        print("未找到任何批次目录！")
        return
    
    total_completed = 0
    total_expected = len(batch_dirs) * 100  # 每个批次100个实验
    active_batches = 0
    
    print(f"{'批次':^8} | {'已完成':^8} | {'进行中':^8} | {'总进度':^8} | {'最后更新时间':^20}")
    print("-" * 70)
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        result_file = os.path.join(batch_dir, "best_results.smi")
        log_file = os.path.join(base_dir, f"{batch_name}_log.txt")
        
        completed = count_experiments_in_file(result_file)
        completed_in_log, started_in_log, last_modified = parse_log_file(log_file)
        
        # 使用两个来源中较大的值
        completed = max(completed, completed_in_log)
        in_progress = max(0, started_in_log - completed)
        
        # 检查是否活跃
        is_active = False
        if last_modified:
            # 如果24小时内有更新，认为是活跃的
            if time.time() - os.path.getmtime(log_file) < 24 * 3600:
                is_active = True
                active_batches += 1
        
        # 计算进度百分比
        progress = completed * 100 / 100  # 每个批次100个实验
        
        # 格式化输出
        status = f"{batch_name:^8} | {completed:^8} | {in_progress:^8} | {progress:^6.1f}% | {last_modified or 'N/A':^20}"
        if is_active:
            status += " (活跃)"
        
        print(status)
        total_completed += completed
    
    # 输出总体进度
    overall_progress = total_completed * 100 / total_expected
    print("\n总体进度:")
    print(f"完成: {total_completed}/{total_expected} ({overall_progress:.1f}%)")
    print(f"活跃批次: {active_batches}/{len(batch_dirs)}")
    
    # 检查运行中的进程
    running_processes = check_running_processes()
    if running_processes >= 0:
        print(f"当前运行中的GA_llm_finetune.py进程: {running_processes}")
    
    # 估计剩余时间
    if total_completed > 0 and active_batches > 0:
        # 假设每个实验平均耗时10分钟
        remaining_experiments = total_expected - total_completed
        estimated_time_minutes = (remaining_experiments * 10) / active_batches
        estimated_hours = estimated_time_minutes / 60
        estimated_days = estimated_hours / 24
        
        print(f"\n估计剩余时间: {estimated_hours:.1f}小时 ({estimated_days:.1f}天)")
        print(f"预计完成时间: {time.ctime(time.time() + estimated_time_minutes * 60)}")

def main():
    parser = argparse.ArgumentParser(description='检查实验进度')
    parser.add_argument('--base_dir', type=str, default='output_3000experiments', help='包含所有批次结果的基础目录')
    
    args = parser.parse_args()
    check_progress(args.base_dir)

if __name__ == "__main__":
    main() 