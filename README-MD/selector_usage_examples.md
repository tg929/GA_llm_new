# AutoGrow4选择器集成使用指南

本文档说明如何在GA_llm_rga.py中使用集成的AutoGrow4选择器功能。

## 新增参数

### `--selector_choice`
指定种子选择器的类型，可选值：
- `Roulette_Selector` (默认): 轮盘赌选择器 - 基于加权概率的随机选择
- `Rank_Selector`: 排名选择器 - 确定性选择，基于分数排名
- `Tournament_Selector`: 锦标赛选择器 - 通过锦标赛机制进行选择

### `--tourn_size`
锦标赛选择器的锦标赛规模 (仅在使用Tournament_Selector时有效)
- 数值范围: 0.0 < tourn_size <= 1.0
- 默认值: 0.1
- 含义: 每个锦标赛参与的分子数量占总分子数量的比例

## 选择器说明

### 1. 轮盘赌选择器 (Roulette_Selector)
- **原理**: 基于分子的适应度分数进行加权随机选择
- **特点**: 
  - 随机性强，能保持种群多样性
  - 分数越好的分子被选中的概率越高
  - 允许较差的分子也有机会被选中
- **适用场景**: 大多数情况下的默认选择

### 2. 排名选择器 (Rank_Selector)
- **原理**: 基于分数排名进行确定性选择，总是选择最好的分子
- **特点**:
  - 确定性选择，无随机性
  - 精英主义策略，总是选择排名靠前的分子
  - 可能导致种群收敛过快，失去多样性
- **适用场景**: 当你想要更激进的选择策略时
- **注意**: 小规模运行时不推荐使用，可能导致种子数量不足

### 3. 锦标赛选择器 (Tournament_Selector)
- **原理**: 随机选择一组分子进行"锦标赛"，选择其中最好的
- **特点**:
  - 平衡了随机性和选择压力
  - 锦标赛规模越大，选择压力越强
  - 可以通过调整tourn_size来控制选择强度
- **适用场景**: 需要精确控制选择压力时

## 使用示例

### 示例1: 使用默认轮盘赌选择器
```bash
python3 GA_llm_rga.py --generations 5 --targets 4r6e
```

### 示例2: 使用排名选择器
```bash
python3 GA_llm_rga.py --generations 5 --targets 4r6e --selector_choice Rank_Selector
```

### 示例3: 使用锦标赛选择器
```bash
python3 GA_llm_rga.py --generations 5 --targets 4r6e --selector_choice Tournament_Selector --tourn_size 0.2
```

### 示例4: 使用较大的锦标赛规模(更强的选择压力)
```bash
python3 GA_llm_rga.py --generations 5 --targets 4r6e --selector_choice Tournament_Selector --tourn_size 0.5
```

## 选择器参数建议

### 锦标赛规模设置建议:
- **0.1 - 0.2**: 较弱的选择压力，保持更多多样性
- **0.3 - 0.5**: 中等选择压力，平衡性能和多样性
- **0.6 - 1.0**: 较强的选择压力，更快收敛但可能失去多样性

### 不同场景的推荐设置:

#### 探索阶段（前几代）:
```bash
--selector_choice Roulette_Selector
# 或
--selector_choice Tournament_Selector --tourn_size 0.1
```

#### 收敛阶段（后几代）:
```bash
--selector_choice Rank_Selector
# 或  
--selector_choice Tournament_Selector --tourn_size 0.5
```

#### 平衡探索和收敛:
```bash
--selector_choice Tournament_Selector --tourn_size 0.3
```

## 技术细节

### 数据格式
选择器使用AutoGrow4标准格式：
- 每个分子包含: [SMILES, NAME, DOCKING_SCORE, DIVERSITY_SCORE]
- 对接分数: 越小越好（更负的值表示更好的结合）
- 多样性分数: 基于分子指纹相似性计算

### 精英保留机制
无论使用哪种选择器，系统都会：
1. 保留每代的最佳分子（精英分子）
2. 在精英分子、适应度种子和多样性种子之间进行去重
3. 确保种子选择的多样性

### 回退机制
如果AutoGrow4选择器模块不可用或出现错误，系统会自动回退到原始的种子选择方法，确保程序的稳定运行。

## 故障排除

### 常见问题:

1. **"AutoGrow4选择器模块不可用"**
   - 检查autogrow目录是否存在
   - 检查Python路径设置
   - 确认相关依赖已安装

2. **"锦标赛规模应在0.0到1.0之间"**
   - 检查--tourn_size参数值
   - 确保数值在有效范围内

3. **Rank_Selector导致种子数量不足**
   - 增加初始种群大小
   - 减少种子选择数量
   - 改用其他选择器

## 性能对比

不同选择器的特性对比：

| 选择器 | 随机性 | 选择压力 | 多样性保持 | 收敛速度 | 计算复杂度 |
|--------|--------|----------|------------|----------|------------|
| Roulette | 高 | 中 | 高 | 中 | 低 |
| Rank | 无 | 高 | 低 | 快 | 低 |
| Tournament | 可调 | 可调 | 可调 | 可调 | 中 |

根据您的具体需求选择合适的选择器和参数设置。 