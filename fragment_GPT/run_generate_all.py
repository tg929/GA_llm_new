import os
import sys
import time
import subprocess

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

start_time = time.time()  # 记录开始时间

for i in range(10):
    cmd = 'python3 generate_all.py' + ' ' + f'--seed {i}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")