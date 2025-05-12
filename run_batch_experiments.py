import os
import sys
import subprocess
import logging
from datetime import datetime
import time

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def setup_logging(output_dir):
    """设置日志"""
    log_file = os.path.join(output_dir, "batch_runner.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("batch_runner")

def run_instance(instance_id, output_dir, logger):
    """运行单个实例"""
    # 创建实例特定的日志文件
    instance_log = os.path.join(output_dir, f"instance_{instance_id}.log")
    
    # 构建命令
    cmd = [
        "python", "run_GA_llm_finetune.py"
    ]
    
    logger.info(f"启动实例 {instance_id}")
    
    try:
        # 使用nohup在后台运行实例
        with open(instance_log, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True  # 确保进程在后台运行
            )
        
        logger.info(f"实例 {instance_id} 已启动，PID: {process.pid}")
        return process.pid
        
    except Exception as e:
        logger.error(f"启动实例 {instance_id} 失败: {str(e)}")
        return None

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "batch_runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("开始批量运行30个实例")
    
    # 存储所有实例的PID
    instance_pids = []
    
    # 启动30个实例
    for i in range(30):
        pid = run_instance(i, output_dir, logger)
        if pid:
            instance_pids.append(pid)
        # 每个实例启动后等待一小段时间，避免同时启动太多进程
        time.sleep(2)
    
    # 保存所有实例的PID到文件
    pid_file = os.path.join(output_dir, "instance_pids.txt")
    with open(pid_file, 'w') as f:
        for i, pid in enumerate(instance_pids):
            f.write(f"Instance {i}: PID {pid}\n")
    
    logger.info(f"已启动 {len(instance_pids)} 个实例")
    logger.info(f"实例PID已保存至: {pid_file}")
    logger.info("所有实例已在后台运行")
    logger.info(f"可以通过以下命令查看实例状态:")
    logger.info(f"ps -p {','.join(map(str, instance_pids))}")
    logger.info(f"可以通过以下命令终止所有实例:")
    logger.info(f"kill {' '.join(map(str, instance_pids))}")

if __name__ == "__main__":
    main() 