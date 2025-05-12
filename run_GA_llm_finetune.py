import os
import sys
import subprocess
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def setup_logging(output_dir):
    """设置日志"""
    log_file = os.path.join(output_dir, "batch_experiments.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("batch_experiments")

def extract_best_molecule(experiment_dir):
    """从一次实验中提取最优分子"""
    best_score = float('inf')
    best_molecule = None
    best_generation = None
    
    # 遍历所有代
    for gen in range(6):  # 0-5代
        docked_file = os.path.join(experiment_dir, f"generation_{gen}", f"generation_{gen}_docked.smi")
        if not os.path.exists(docked_file):
            continue
            
        # 读取对接结果
        with open(docked_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        molecule = parts[0]
                        score = float(parts[1])
                        if score < best_score:
                            best_score = score
                            best_molecule = molecule
                            best_generation = gen
    
    return best_molecule, best_score, best_generation

def run_experiment(experiment_id, output_base_dir, logger):
    """运行单次实验"""
    # 创建实验目录
    experiment_dir = os.path.join(output_base_dir, f"experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        "python", "GA_llm_finetune.py",
        "--output_dir", experiment_dir,
        "--generations", "5"
    ]
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"开始实验 {experiment_id}")
    
    try:
        # 运行实验
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"实验 {experiment_id} 失败: {process.stderr}")
            return None
        
        # 提取最优分子
        best_molecule, best_score, best_generation = extract_best_molecule(experiment_dir)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"实验 {experiment_id} 完成，耗时: {duration:.2f}秒")
        logger.info(f"最优分子: {best_molecule}")
        logger.info(f"最优得分: {best_score}")
        logger.info(f"最优代: {best_generation}")
        
        return {
            'experiment_id': experiment_id,
            'best_molecule': best_molecule,
            'best_score': best_score,
            'best_generation': best_generation,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"实验 {experiment_id} 出错: {str(e)}")
        return None

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(PROJECT_ROOT, "batch_experiments", timestamp)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_base_dir)
    logger.info("开始批量实验")
    
    # 运行100次实验
    results = []
    for i in range(100):
        result = run_experiment(i, output_base_dir, logger)
        if result:
            results.append(result)
    
    # 将结果保存为CSV文件
    if results:
        df = pd.DataFrame(results)
        csv_file = os.path.join(output_base_dir, "experiment_results.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"实验结果已保存至: {csv_file}")
        
        # 输出统计信息
        logger.info("\n实验统计信息:")
        logger.info(f"总实验数: {len(results)}")
        logger.info(f"平均运行时间: {df['duration'].mean():.2f}秒")
        logger.info(f"最优得分: {df['best_score'].min():.4f}")
        logger.info(f"平均最优得分: {df['best_score'].mean():.4f}")
        logger.info(f"标准差: {df['best_score'].std():.4f}")
        
        # 保存最优分子到单独的文件
        best_molecules_file = os.path.join(output_base_dir, "best_molecules.smi")
        with open(best_molecules_file, 'w') as f:
            for _, row in df.iterrows():
                f.write(f"{row['best_molecule']}\n")
        logger.info(f"最优分子已保存至: {best_molecules_file}")
    
    logger.info("批量实验完成")

if __name__ == "__main__":
    main()
