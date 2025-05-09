import os
import time
import subprocess
import sys

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

start_time = time.time()  # 记录开始时间

for i in range(30):
    if i < 15:
        device = 2
    else:
        device = 3
    cmd = 'python3 mcts_new_weight_40_experiment.py' + ' ' + '--output_file_path' + ' ' + f'./output/mcts_20_new_weight/mcts_5ht1b_20_new_{i}.csv' + ' '\
                                         + f'--device {device}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")

# 执行命令
os.system("python mcts_decode_new_40.py")