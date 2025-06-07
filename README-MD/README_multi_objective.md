# GA_llm_rga NSGA-II帕累托多目标优化功能说明

## 概述

GA_llm_rga现在采用NSGA-II帕累托算法进行真正的多目标优化，同时优化对接分数、QED分数和SA分数，不再使用单目标综合评分。

## 主要特性

### 1. NSGA-II帕累托多目标优化

采用真正的多目标优化算法：

**优化目标：**
- **对接分数**：最小化（越小越好）
- **QED分数**：最大化（药物相似性）
- **SA分数**：最小化（合成难度越小越好）

**选择策略：**
- 帕累托前沿识别
- 多策略分子选择（对接优先、QED优先、SA优先、综合评分）
- 自动回退机制

### 2. 真正的多目标选择

- **帕累托前沿**：找到在三个目标上都不被支配的解
- **多样化选择**：从帕累托前沿中按不同策略选择分子
- **精英保留**：确保最优分子不会丢失

### 3. 智能回退机制

- **主要方法**：NSGA-II帕累托算法选择
- **备用方法**：基于对接分数的简单选择
- **自动切换**：当帕累托算法失败时自动回退

## 使用方法

### 基本使用

```bash
python GA_llm_rga.py --generations 5 --targets 4r6e
```

### 多受体并行优化

```bash
python GA_llm_rga.py --targets 4r6e 3pbl 1iep --parallel --max_workers 4
```

### 自定义种子选择参数

```bash
python GA_llm_rga.py --targets 4r6e \
    --top_mols_to_seed_next_generation 15 \
    --diversity_mols_to_seed_first_generation 15 \
    --generations 10
```

### 推荐的完整命令

```bash
python GA_llm_rga.py \
    --output_dir output_rga \
    --targets 4r6e 3pbl \
    --top_mols_to_seed_next_generation 10 \
    --diversity_mols_to_seed_first_generation 10 \
    --diversity_seed_depreciation_per_gen 2 \
    --parallel \
    --max_workers 4
```

## 输出变化

### 评估指标文件

在`generation_X_evaluation_metrics.txt`文件中，现在包含：

```
Docking Score - Top 1: -10.3000
QED - Top 100 Mean: 0.5543
SA Score - Top 100 Mean: 2.2366
Novelty: 0.8500
Diversity (Internal): 0.7234
Diversity (Top 100): 0.6891
```

### NSGA-II选择日志

种子选择过程中会显示：
- 帕累托前沿分子数量
- 选择策略统计
- 精英分子保留信息

## 技术细节

### NSGA-II算法流程

1. **分子评估**：计算每个分子的对接分数、QED分数、SA分数
2. **帕累托前沿识别**：找到在三个目标上都不被支配的解
3. **多策略选择**：
   - 对接优先：按对接分数排序选择
   - QED优先：按QED分数排序选择
   - SA优先：按SA分数排序选择
   - 综合评分：按加权组合排序选择
4. **精英保留**：确保当前最优分子被保留

### 目标函数设计

- **f1 = 对接分数**（最小化）
- **f2 = -QED分数**（最小化，即最大化QED）
- **f3 = SA分数**（最小化）

### 选择机制

- 如果帕累托前沿分子足够：按不同策略分配选择
- 如果帕累托前沿分子不足：选择全部帕累托分子，再补充其他优秀分子
- 自动去重和精英分子保留

## 性能特点

### 优势
- **真正的多目标优化**：不需要人工权重设定
- **帕累托最优**：找到真正的最优解集
- **多样化解**：获得多种不同特性的分子
- **自动平衡**：在多个目标间自动平衡

### 与单目标对比
- 不需要手动调整权重
- 避免了线性组合的局限性
- 得到更多样化的解集
- 更好的目标空间覆盖

## 故障排除

### 常见问题

1. **NSGA-II选择失败**：会自动回退到简单选择方法
2. **pymoo库缺失**：将使用简化的帕累托算法
3. **分子转换失败**：自动过滤无效分子

### 调试建议

- 检查日志中的帕累托选择信息
- 验证评估文件中的多目标指标
- 对比不同代数的帕累托前沿演化

## 依赖库

- **pymoo**：用于NSGA-II算法（可选）
- **rdkit**：用于分子处理和QED计算
- **numpy**：用于数值计算

如果pymoo不可用，系统会自动使用简化的帕累托算法。

## 向后兼容性

- 保持所有原有参数接口
- 评估指标仍然计算QED和SA的单独统计
- 日志格式基本保持不变，增加了帕累托选择信息 