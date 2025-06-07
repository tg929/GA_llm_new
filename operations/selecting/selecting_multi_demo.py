#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于NSGA-II帕累托算法的多目标分子选择脚本
同时优化对接分数、QED分数和SA分数

目标优化方向：
- 对接分数：最小化（越小越好）
- QED分数:最大化(转换为最小化 -QED)
- SA分数:最小化(越小越好)
"""

import argparse
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 导入SA计算功能
try:
    from fragment_GPT.utils import sascorer
    SA_CALCULATOR = sascorer.calculateScore
except ImportError:
    try:
        import sascorer
        SA_CALCULATOR = sascorer.calculateScore
    except ImportError:
        print("警告: 无法导入SA计算模块，SA分数将设为默认值")
        SA_CALCULATOR = None

# 导入pymoo
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    import pymoo
    print(f"成功导入pymoo版本: {pymoo.__version__}")
except ImportError:
    print("错误: 未找到pymoo库。请运行 'pip install pymoo' 安装。")
    sys.exit(1)

class MolecularSelectionProblem(Problem):
    """
    分子选择的多目标优化问题定义
    
    目标函数：
    1. 最小化对接分数（越小越好）
    2. 最小化 -QED(即最大化QED,越大越好)
    3. 最小化SA分数(越小越好)
    """
    
    def __init__(self, molecules_data):
        """
        初始化问题
        
        Args:
            molecules_data: list of dict, 每个dict包含:
                - 'smiles': SMILES字符串
                - 'docking_score': 对接分数
                - 'qed_score': QED分数 
                - 'sa_score': SA分数
        """
        self.molecules_data = molecules_data
        self.n_molecules = len(molecules_data)
        
        # 定义问题：n_var = 分子数量，n_obj = 3个目标，变量类型为二进制（选择或不选择）
        super().__init__(
            n_var=self.n_molecules,  # 每个分子一个二进制变量
            n_obj=3,                 # 3个目标函数
            n_constr=0,              # 无约束
            xl=0,                    # 变量下界
            xu=1,                    # 变量上界
            type_var=int             # 整数变量（0或1）
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估目标函数
        
        Args:
            X: 解矩阵，每行代表一个解（分子选择方案）
            out: 输出字典，包含目标函数值
        """
        F = []  # 目标函数值矩阵
        
        for solution in X:
            # solution是一个二进制数组，表示哪些分子被选择
            selected_indices = np.where(solution == 1)[0]
            
            if len(selected_indices) == 0:
                # 如果没有选择任何分子，给予惩罚值
                f1 = 1000  # 对接分数惩罚
                f2 = 1000  # QED惩罚（-QED的惩罚）
                f3 = 1000  # SA分数惩罚
            else:
                # 计算选中分子的目标函数值
                selected_docking = [self.molecules_data[i]['docking_score'] for i in selected_indices]
                selected_qed = [self.molecules_data[i]['qed_score'] for i in selected_indices]
                selected_sa = [self.molecules_data[i]['sa_score'] for i in selected_indices]
                
                # 目标1: 最小化对接分数的均值
                f1 = np.mean(selected_docking)
                
                # 目标2: 最小化 -QED（即最大化QED）
                f2 = -np.mean(selected_qed)
                
                # 目标3: 最小化SA分数的均值
                f3 = np.mean(selected_sa)
            
            F.append([f1, f2, f3])
        
        out["F"] = np.array(F)

def load_molecules_with_scores(docked_file):
    """
    从对接结果文件中加载分子及其分数
    
    Args:
        docked_file: 对接结果文件路径，格式为 "SMILES score"
    
    Returns:
        list: 分子数据列表,每个元素包含SMILES和对接分数
    """
    molecules = []
    
    try:
        with open(docked_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        smiles = parts[0]
                        try:
                            docking_score = float(parts[1])
                            molecules.append({
                                'smiles': smiles,
                                'docking_score': docking_score
                            })
                        except ValueError:
                            print(f"警告: 无法解析分数 {parts[1]} for SMILES {smiles}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {docked_file}")
        return []
    
    return molecules

def calculate_qed_score(smiles):
    """计算QED分数"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return QED.qed(mol)
    except:
        return 0.0

def calculate_sa_score(smiles):
    """计算SA分数"""
    if SA_CALCULATOR is None:
        return 5.0  # 默认中等难度
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0  # 最差SA分数
        return SA_CALCULATOR(mol)
    except:
        return 10.0

def add_additional_scores(molecules):
    """为分子添加QED和SA分数"""
    print("正在计算QED和SA分数...")
    
    for i, mol_data in enumerate(molecules):
        smiles = mol_data['smiles']
        
        # 计算QED分数
        qed_score = calculate_qed_score(smiles)
        mol_data['qed_score'] = qed_score
        
        # 计算SA分数
        sa_score = calculate_sa_score(smiles)
        mol_data['sa_score'] = sa_score
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(molecules)} 个分子")
    
    print(f"完成所有 {len(molecules)} 个分子的分数计算")
    return molecules

def create_initial_population(n_molecules, population_size, selection_pressure=0.3):
    """
    创建初始种群，偏向选择更好的分子
    
    Args:
        n_molecules: 总分子数
        population_size: 种群大小
        selection_pressure: 选择压力，越高越倾向于选择排名靠前的分子
    
    Returns:
        初始种群矩阵
    """
    population = []
    
    for _ in range(population_size):
        # 创建一个解（分子选择方案）
        solution = np.zeros(n_molecules, dtype=int)
        
        # 根据选择压力决定选择多少个分子
        # 选择10-50个分子（可调整）
        n_select = np.random.randint(10, min(50, n_molecules))
        
        # 使用加权随机选择，偏向前面的分子（假设已按对接分数排序）
        weights = np.exp(-selection_pressure * np.arange(n_molecules))
        probabilities = weights / np.sum(weights)
        
        selected_indices = np.random.choice(
            n_molecules, 
            size=n_select, 
            replace=False, 
            p=probabilities
        )
        
        solution[selected_indices] = 1
        population.append(solution)
    
    return np.array(population)

def select_molecules_nsga2(molecules_data, n_select_fitness=50, n_select_diversity=25, 
                          population_size=100, generations=50):
    """
    使用NSGA-II算法进行多目标分子选择
    
    Args:
        molecules_data: 分子数据列表
        n_select_fitness: 基于适应度选择的分子数量
        n_select_diversity: 基于多样性选择的分子数量  
        population_size: NSGA-II种群大小（用于兼容性，实际上使用简化方法）
        generations: NSGA-II进化代数（用于兼容性，实际上使用简化方法）
    
    Returns:
        selected_molecules: 选中的分子列表
        pareto_front: 帕累托前沿解集
    """
    print(f"开始多目标选择: {len(molecules_data)} 个分子")
    print(f"目标: 选择 {n_select_fitness} 个适应度分子 + {n_select_diversity} 个多样性分子")
    
    if len(molecules_data) == 0:
        return [], []
    
    total_select = n_select_fitness + n_select_diversity
    
    # 方法1: 简化的帕累托前沿方法
    # 创建目标函数矩阵
    objectives = []
    for mol in molecules_data:
        f1 = mol['docking_score']  # 最小化（越小越好）
        f2 = -mol['qed_score']     # 最小化-QED（即最大化QED）
        f3 = mol['sa_score']       # 最小化（越小越好）
        objectives.append([f1, f2, f3])
    
    objectives = np.array(objectives)
    
    # 找到帕累托最优解
    print("计算帕累托前沿...")
    pareto_indices = find_pareto_front(objectives)
    print(f"找到 {len(pareto_indices)} 个帕累托最优分子")
    
    selected_molecules = []
    used_indices = set()
    
    # 策略1: 从帕累托前沿中选择
    pareto_molecules = [molecules_data[i] for i in pareto_indices]
    
    if len(pareto_molecules) >= total_select:
        # 如果帕累托前沿分子足够，按不同策略选择
        strategies = [
            ('对接优先', lambda mol: mol['docking_score']),
            ('QED优先', lambda mol: -mol['qed_score']),  # 负号因为要最大化QED
            ('SA优先', lambda mol: mol['sa_score']),
            ('综合评分', lambda mol: 0.4*mol['docking_score'] - 0.3*mol['qed_score'] + 0.3*mol['sa_score'])
        ]
        
        molecules_per_strategy = total_select // len(strategies)
        
        for strategy_name, sort_key in strategies:
            if len(selected_molecules) >= total_select:
                break
                
            print(f"使用 '{strategy_name}' 策略选择分子")
            
            # 对帕累托分子按当前策略排序
            sorted_pareto = sorted(
                [(i, mol) for i, mol in enumerate(pareto_molecules) if i not in used_indices],
                key=lambda x: sort_key(x[1])
            )
            
            # 选择当前策略的分子
            strategy_select = min(molecules_per_strategy, len(sorted_pareto))
            for i in range(strategy_select):
                if len(selected_molecules) >= total_select:
                    break
                    
                idx, mol = sorted_pareto[i]
                selected_molecules.append(mol)
                used_indices.add(idx)
        
        # 如果还需要更多分子，随机从剩余的帕累托分子中选择
        if len(selected_molecules) < total_select:
            remaining_pareto = [(i, mol) for i, mol in enumerate(pareto_molecules) if i not in used_indices]
            np.random.shuffle(remaining_pareto)
            
            for idx, mol in remaining_pareto:
                if len(selected_molecules) >= total_select:
                    break
                selected_molecules.append(mol)
                used_indices.add(idx)
    
    else:
        # 如果帕累托前沿分子不够，先选择全部帕累托分子，再补充其他优秀分子
        print(f"帕累托前沿分子数量({len(pareto_molecules)})少于需求({total_select})，添加额外分子")
        
        # 添加所有帕累托分子
        selected_molecules.extend(pareto_molecules)
        used_original_indices = set(pareto_indices)
        
        # 从非帕累托分子中选择补充
        remaining_molecules = [(i, mol) for i, mol in enumerate(molecules_data) if i not in used_original_indices]
        
        # 按综合评分排序剩余分子
        remaining_molecules.sort(key=lambda x: 0.4*x[1]['docking_score'] - 0.3*x[1]['qed_score'] + 0.3*x[1]['sa_score'])
        
        needed = total_select - len(selected_molecules)
        for i in range(min(needed, len(remaining_molecules))):
            selected_molecules.append(remaining_molecules[i][1])
    
    print(f"多目标选择完成，共选择 {len(selected_molecules)} 个分子")
    
    # 创建帕累托前沿信息
    pareto_info = {
        'pareto_indices': pareto_indices,
        'objectives': objectives[pareto_indices],
        'n_solutions': len(pareto_indices)
    }
    
    return selected_molecules, pareto_info

def find_pareto_front(objectives):
    """
    找到帕累托前沿
    
    Args:
        objectives: 目标函数矩阵，每行是一个解的目标函数值
    
    Returns:
        pareto_indices: 帕累托最优解的索引列表
    """
    n_points = objectives.shape[0]
    pareto_indices = []
    
    for i in range(n_points):
        is_pareto = True
        for j in range(n_points):
            if i != j:
                # 检查是否被支配（所有目标都不比j差，且至少一个目标比j差）
                if all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices

def select_molecules_fallback(molecules_data, n_select):
    """
    备选的简单选择方案（当NSGA-II失败时使用）
    """
    print(f"使用备选选择方案，选择前 {n_select} 个分子")
    
    # 按对接分数排序
    sorted_molecules = sorted(molecules_data, key=lambda x: x['docking_score'])
    
    return sorted_molecules[:n_select]

def save_selected_molecules(selected_molecules, output_file):
    """保存选中的分子到文件"""
    with open(output_file, 'w') as f:
        for mol_data in selected_molecules:
            f.write(f"{mol_data['smiles']}\n")
    
    print(f"已保存 {len(selected_molecules)} 个选中的分子到 {output_file}")

def print_selection_statistics(selected_molecules):
    """打印选择统计信息"""
    if not selected_molecules:
        print("没有选择任何分子")
        return
    
    docking_scores = [mol['docking_score'] for mol in selected_molecules]
    qed_scores = [mol['qed_score'] for mol in selected_molecules]
    sa_scores = [mol['sa_score'] for mol in selected_molecules]
    
    print("\n========== 选择统计信息 ==========")
    print(f"选中分子数量: {len(selected_molecules)}")
    print(f"对接分数 - 最优: {min(docking_scores):.4f}, 平均: {np.mean(docking_scores):.4f}")
    print(f"QED分数 - 最优: {max(qed_scores):.4f}, 平均: {np.mean(qed_scores):.4f}")
    print(f"SA分数 - 最优: {min(sa_scores):.4f}, 平均: {np.mean(sa_scores):.4f}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description='基于NSGA-II的多目标分子选择')
    
    # 输入输出参数
    parser.add_argument('--docked_file', type=str, required=True,
                       help='对接结果文件路径（格式: SMILES score）')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出的种子分子文件路径')
    
    # 选择参数
    parser.add_argument('--n_select_fitness', type=int, default=50,
                       help='基于适应度选择的分子数量')
    parser.add_argument('--n_select_diversity', type=int, default=25,
                       help='基于多样性选择的分子数量')
    
    # NSGA-II参数
    parser.add_argument('--population_size', type=int, default=100,
                       help='NSGA-II种群大小')
    parser.add_argument('--generations', type=int, default=50,
                       help='NSGA-II进化代数')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    print("开始基于NSGA-II的多目标分子选择...")
    print(f"输入文件: {args.docked_file}")
    print(f"输出文件: {args.output_file}")
    
    # 1. 加载分子及对接分数
    molecules = load_molecules_with_scores(args.docked_file)
    if not molecules:
        print("错误: 无法加载分子数据")
        return
    
    print(f"加载了 {len(molecules)} 个分子")
    
    # 2. 计算QED和SA分数
    molecules = add_additional_scores(molecules)
    
    # 3. 使用NSGA-II进行多目标选择
    selected_molecules, pareto_info = select_molecules_nsga2(
        molecules,
        n_select_fitness=args.n_select_fitness,
        n_select_diversity=args.n_select_diversity,
        population_size=args.population_size,
        generations=args.generations
    )
    
    # 4. 保存结果
    if selected_molecules:
        save_selected_molecules(selected_molecules, args.output_file)
        print_selection_statistics(selected_molecules)
        
        if args.verbose and pareto_info['objectives'] is not None:
            print(f"\n帕累托前沿包含 {pareto_info['n_solutions']} 个解")
            print("目标函数统计 (对接分数, -QED, SA分数):")
            objectives = pareto_info['objectives']
            for i in range(3):
                obj_values = objectives[:, i]
                print(f"目标{i+1}: 最小值={np.min(obj_values):.4f}, 平均值={np.mean(obj_values):.4f}, 最大值={np.max(obj_values):.4f}")
    else:
        print("错误: 未选择任何分子")

if __name__ == "__main__":
    main()
