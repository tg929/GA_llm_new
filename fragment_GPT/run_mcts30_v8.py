import os
import time
import subprocess
import sys

start_time = time.time()  # 记录开始时间

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

for i in range(60):
    if i < 15:
        device = 9
    elif 15 <= i < 30:
        device = 7
    elif 30 <= i < 45:
        device = 6
    else:
        device = 5
    cmd = 'python3 mcts_experiment_v8.py' + ' ' + '--output_file_path' + ' ' + f'./output/mcts_v8_new_weight/mcts_new_5ht1b_v8_{i}.csv' + ' '\
                                          + f'--device {device}' + ' ' + f'--seed {i}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")

# 执行命令
os.system("python mcts_decode_v8.py")