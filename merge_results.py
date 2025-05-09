#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import argparse
from pathlib import Path

def merge_results(base_dir="output_3000experiments", output_file="all_best_results.smi"):
    """合并所有批次的最佳结果文件"""
    print("开始合并结果...")
    
    # 查找所有结果文件
    result_files = []
    for batch_dir in glob.glob(os.path.join(base_dir, "batch_*")):
        result_file = os.path.join(batch_dir, "best_results.smi")
        if os.path.exists(result_file):
            result_files.append(result_file)
    
    print(f"找到 {len(result_files)} 个批次结果文件")
    
    # 读取所有结果
    all_results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            smiles = parts[0]
                            score = float(parts[1])
                            info = parts[2] if len(parts) > 2 else ""
                            all_results.append((smiles, score, info))
        except Exception as e:
            print(f"读取文件 {result_file} 时出错: {str(e)}")
    
    print(f"总共读取了 {len(all_results)} 个结果")
    
    # 按得分排序 (得分越低越好)
    all_results.sort(key=lambda x: x[1])
    
    # 保存排序后的结果
    with open(output_file, 'w') as f:
        for smiles, score, info in all_results:
            f.write(f"{smiles}\t{score:.4f}\t{info}\n")
    
    print(f"已将所有结果合并并排序保存至 {output_file}")
    
    # 输出一些统计信息
    if all_results:
        scores = [result[1] for result in all_results]
        print(f"最佳得分: {min(scores):.4f}")
        print(f"平均得分: {np.mean(scores):.4f}")
        print(f"得分中位数: {np.median(scores):.4f}")
        print(f"前10个最佳结果:")
        for i, (smiles, score, info) in enumerate(all_results[:10]):
            print(f"{i+1}. 得分: {score:.4f}, 来源: {info}, SMILES: {smiles[:50]}...")

def main():
    parser = argparse.ArgumentParser(description='合并所有批次的实验结果')
    parser.add_argument('--base_dir', type=str, default='output_3000experiments', help='包含所有批次结果的基础目录')
    parser.add_argument('--output', type=str, default='all_best_results.smi', help='合并后的输出文件')
    
    args = parser.parse_args()
    merge_results(args.base_dir, args.output)

if __name__ == "__main__":
    main() 