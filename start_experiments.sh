#!/bin/bash

# 创建总输出目录
mkdir -p output_3000experiments

# 设置环境变量，限制Python使用的线程数
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# 设置每个批次使用的CPU核心数
CORES_PER_BATCH=4

# 减少并行批次数量,从10改为5,进一步降低资源竞争
MAX_PARALLEL=5
TOTAL_BATCHES=30

echo "准备启动 $TOTAL_BATCHES 个批次，每次最多并行 $MAX_PARALLEL 个批次"
echo "每个批次将使用 $CORES_PER_BATCH 个CPU核心"

# 创建日志目录
mkdir -p output_3000experiments/logs

# 创建一个临时脚本目录
mkdir -p output_3000experiments/scripts

# 按组运行批次
for group in $(seq 0 $((MAX_PARALLEL-1)) $((TOTAL_BATCHES-1)))
do
    echo "启动批次组 $((group/MAX_PARALLEL+1))..."
    
    # 在每组中启动最多MAX_PARALLEL个批次
    for i in $(seq $group $((group+MAX_PARALLEL-1)))
    do
        # 确保不超过总批次数
        if [ $i -lt $TOTAL_BATCHES ]; then
            start_id=$((i * 100))
            output_dir="output_3000experiments/batch_$i"
            
            echo "启动批次 $i (实验 $start_id-$((start_id + 99)))"
            
            # 为该批次创建单独的脚本,避免直接在shell中运行复杂命令
            script_file="output_3000experiments/scripts/run_batch_${i}.sh"
            
            # 写入脚本内容
            cat > $script_file << EOF
#!/bin/bash
echo "开始批次 $i (实验 $start_id-$((start_id+99)))"
export OMP_NUM_THREADS=$CORES_PER_BATCH
export OPENBLAS_NUM_THREADS=$CORES_PER_BATCH
export MKL_NUM_THREADS=$CORES_PER_BATCH
export PYTHONPATH=\${PYTHONPATH}:$(pwd)

# 确保输出目录存在
mkdir -p $output_dir

# 运行批处理
python run_GA_llm_finetune.py \\
    --start_id $start_id \\
    --num_experiments 100 \\
    --output_dir $output_dir \\
    --number_of_processors $CORES_PER_BATCH

echo "批次 $i 完成"
EOF
            
            # 设置脚本执行权限
            chmod +x $script_file
            
            # 使用nohup在后台运行脚本
            nohup bash $script_file > "output_3000experiments/logs/batch_${i}_log.txt" 2>&1 &
            
            # 记录进程ID
            echo "批次 $i 启动,PID: $!"
            
            # 增加延迟，避免同时启动太多进程
            sleep 10
        fi
    done
    
    # 等待这组批次完成一些工作后再启动下一组
    echo "等待60秒后启动下一组批次..."
    sleep 60
done

echo "已启动所有 $TOTAL_BATCHES 个批次,总共3000个实验"
echo "可以使用 'ps aux | grep run_GA_llm_finetune.py' 查看运行状态"
echo "使用 'python check_progress.py --base_dir output_3000experiments' 监控进度"

# 创建一个简单的状态监控脚本
cat > output_3000experiments/check_status.sh << EOF
#!/bin/bash
echo "正在运行的批次:"
ps aux | grep "run_batch_" | grep -v grep

echo -e "\n已完成的批次数量:"
find output_3000experiments/batch_* -name "best_results.smi" 2>/dev/null | wc -l

echo -e "\n各批次情况:"
for i in \$(seq 0 $((TOTAL_BATCHES-1))); do
    COUNT=\$(find output_3000experiments/batch_\$i -name "exp_*" 2>/dev/null | wc -l)
    echo "批次 \$i: 已完成 \$COUNT/100 个实验"
done
EOF

chmod +x output_3000experiments/check_status.sh
echo "可以使用 './output_3000experiments/check_status.sh' 查看批次运行状态" 