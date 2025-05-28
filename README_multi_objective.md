# GA_llm_rga 多目标优化功能说明

## 概述

本次更新为GA_llm_rga添加了多目标优化功能，解决了之前仅基于对接分数的单目标优化导致QED和SA分数过低的问题。

## 主要改进

### 1. 多目标综合评分Y

根据论文公式实现了综合评分计算：

```
y = DS × QED × SA ∈ [0, 1]
```

其中：
- **DS** = -clip(DS)/20 ∈ [0, 1] （标准化对接分数）
- **QED** ∈ [0, 1] （药物相似性分数）
- **SA** = (10 - SA)/9 ∈ [0, 1] （标准化合成可达性分数）

### 2. 智能种子选择

- **多目标模式**：基于综合评分Y选择种子分子，平衡对接性能、药物相似性和合成可达性
- **单目标模式**：传统的仅基于对接分数的选择方式（向后兼容）
- **自动回退**：当多目标计算失败时，自动回退到单目标模式

### 3. 增强的评估指标

在评估文件中新增：
- **综合评分Y的统计信息**：Top 1、Top 10、Top 100、全体均值
- **仅计算top 100分子的QED和SA**：替代原来的全体种群计算
- **Top 100多样性**：基于对接分数排序后的top分子计算多样性

## 使用方法

### 启用多目标优化（默认）

```bash
python GA_llm_rga.py --generations 5 --targets 1iep --use_multi_objective
```

### 强制使用单目标优化

```bash
python GA_llm_rga.py --generations 5 --targets 1iep --use_single_objective
```

### 推荐的完整命令

```bash
python GA_llm_rga.py \
    --output_dir output_rga \
    --top_mols_to_seed_next_generation 10 \
    --diversity_mols_to_seed_first_generation 10 \
    --diversity_seed_depreciation_per_gen 2 \
    --LipinskiStrictFilter \
    --parallel \
    --use_multi_objective
```

## 新增命令行参数

- `--use_multi_objective`：启用多目标优化（默认启用）
- `--use_single_objective`：强制使用单目标优化

## 输出变化

### 评估指标文件变化

在`generation_X_evaluation_metrics.txt`文件中，现在包含：

```
Y Score (Multi-objective) - Top 1: 0.1234
Y Score (Multi-objective) - Top 10 Mean: 0.0987
Y Score (Multi-objective) - Top 100 Mean: 0.0765
Y Score (Multi-objective) - All Mean: 0.0543
--------------------------------------------------
Formula: Y = DS_normalized × QED × SA_normalized
Where: DS_norm = -clip(DS)/20, SA_norm = (10-SA)/9
```

### 日志输出增强

种子选择过程中会显示：
- 是否使用多目标优化
- Top 5 Y分数值
- 选择的优化方式说明

## 技术细节

### 计算过程

1. **读取对接结果**：从docking output文件读取分子和对接分数
2. **RDKit转换**：将SMILES转换为RDKit分子对象
3. **QED计算**：使用RDKit计算药物相似性分数
4. **SA计算**：使用sascorer计算合成可达性分数
5. **标准化**：将所有分数标准化到[0,1]范围
6. **综合评分**：计算Y = DS × QED × SA
7. **排序选择**：按Y分数降序排序选择种子

### 错误处理

- 分子转换失败时自动过滤无效分子
- QED/SA计算失败时自动回退到单目标模式
- 导入错误时提供详细错误信息

### 性能优化

- 只对有对接分数的分子计算QED和SA
- 缓存计算结果避免重复计算
- 智能长度对齐确保数据一致性

## 验证方法

检查评估文件中的Y分数统计：
- Y分数应该在合理范围内（通常0.01-0.5）
- Top分子的Y分数应该明显高于平均值
- QED和SA分数应该比之前的版本有所改善

## 故障排除

### 常见问题

1. **RDKit导入失败**：确保已安装RDKit库
2. **SA计算失败**：检查sascorer模块是否可用
3. **多目标计算失败**：会自动回退到单目标模式，检查日志获取详细信息

### 调试建议

- 查看日志文件中的详细错误信息
- 检查评估文件中是否包含Y分数统计
- 对比单目标和多目标模式的结果差异

## 向后兼容性

- 所有原有参数和功能保持不变
- 默认启用多目标优化，但可以通过参数关闭
- 原有的评估指标仍然计算和输出 