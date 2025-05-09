import os
import time
import subprocess
import sys

start_time = time.time()  # 记录开始时间

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

for i in range(30):
    device = 1
    cmd = 'python3 mcts_experiment_v7.py' + ' ' + '--output_file_path' + ' ' + f'./output/mcts_v7/mcts_5ht1b_v7_{i}.csv' + ' '\
                                         + f'--device {device}' + ' ' + f'--seed {i}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")

# 执行命令
os.system("python mcts_decode.py")