import os
import sys
import glob
import subprocess
import time
import re
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
import argparse
from typing import Dict, Optional, Tuple
# 导入必要的模块
from autogrow.operators.convert_files.conversion_to_3d import convert_to_3d
from autogrow.operators.convert_files.conversion_to_3d import convert_sdf_to_pdbs
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion
from autogrow.docking.docking_class.docking_file_conversion.convert_with_obabel import ObabelConversion
from autogrow.operators.convert_files.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer

from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from autogrow.docking.docking_class.docking_class_children.quick_vina_2_docking import QuickVina2Docking

from autogrow.docking.execute_docking import run_docking_common
from autogrow.docking.docking_class.parent_pdbqt_converter import ParentPDBQTConverter
from autogrow.docking.docking_class.parent_dock_class import ParentDocking
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def vina_dock_single(ligand_file, receptor_pdbqt, results_dir, vars):
    import subprocess, os
    out_file = os.path.join(
        results_dir,
        os.path.basename(ligand_file).replace(".pdbqt", "_out.pdbqt")
    )
    log_file = os.path.join(
        results_dir,
        os.path.basename(ligand_file).replace(".pdbqt", ".log")
    )
    cmd = [
        vars["docking_executable"],
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_file,
        "--center_x", str(vars["center_x"]),
        "--center_y", str(vars["center_y"]),
        "--center_z", str(vars["center_z"]),
        "--size_x", str(vars["size_x"]),
        "--size_y", str(vars["size_y"]),
        "--size_z", str(vars["size_z"]),
        "--out", out_file,
        "--log", log_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 读取分数
        score = extract_vina_score(log_file)
        # 删除log文件
        if os.path.exists(log_file):
            os.remove(log_file)
        return ligand_file, True, score
    except Exception as e:
        return ligand_file, False, "NA"


def keep_one_pdb_per_smiles(pdb_dir):
    """
    只保留每个SMILES编号的第一个PDB文件,其余全部删除
    """
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    seen = set()
    pattern = re.compile(r"(naphthalene_\d+)")
    for pdb_file in pdb_files:
        base = os.path.basename(pdb_file)
        # 提取SMILES编号（如naphthalene_1）
        m = pattern.search(base)
        if m:
            key = m.group(1)
            if key in seen:
                os.remove(pdb_file)
            else:
                seen.add(key)
def extract_vina_score(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if line.strip().startswith("1"):
                parts = line.split()
                if len(parts) > 1:
                    return parts[1]  # affinity
    return "NA"

def output_smiles_scores(smiles_file, scores_dict, output_file):
    # 读取SMILES与分子名映射
    smiles_map = {}
    with open(smiles_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_map[parts[1]] = parts[0]
    with open(output_file, "w") as out:
        out.write("smiles\tscore\n")
        for mol_name, smiles in smiles_map.items():
            score = scores_dict.get(mol_name, "NA")
            out.write(f"{smiles}\t{score}\n")



class DockingWorkflow:
    """分子对接工作流程类"""
    
    def __init__(self, config: Dict):
        """
        初始化工作流程
        
        参数:
        :param dict config: 配置参数字典
        """
        self.vars = config
        self.setup_directories()
        
    def setup_directories(self):
        """创建必要的目录结构"""
        # 创建输出目录
        if not os.path.exists(self.vars["output_directory"]):
            os.makedirs(self.vars["output_directory"])
            
        # 创建临时文件目录
        self.temp_dir = os.path.join(self.vars["output_directory"], "temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
    def prepare_receptor(self) -> str:
        """
        准备受体蛋白文件
        
        返回:
        :returns: str: 处理后的受体文件路径
        """
        print("准备受体蛋白...")
        
        # 选择文件转换类
        file_conversion_class = self.pick_conversion_class(self.vars["conversion_choice"])
        file_converter = file_conversion_class(
            self.vars,
            self.vars["filename_of_receptor"],
            test_boot=False
        )
        
        # 转换受体文件
        receptor_pdbqt = self.vars["filename_of_receptor"] + "qt"
        if not os.path.exists(receptor_pdbqt):
            file_converter.convert_receptor_pdb_files_to_pdbqt(
                self.vars["filename_of_receptor"],
                self.vars["mgl_python"],
                self.vars["prepare_receptor4.py"],
                self.vars["number_of_processors"]
            )
            
        return receptor_pdbqt
        
    def prepare_ligands(self, smi_file: str) -> str:
        print("准备配体分子...")

        # 1. SMILES转3D SDF
        ligand_dir = self.vars["ligand_dir"]
        if not os.path.exists(ligand_dir):
            os.makedirs(ligand_dir)
        convert_to_3d(self.vars, smi_file, ligand_dir)

        # 2. SDF转PDB
        sdf_dir = self.vars["sdf_dir"]
        convert_sdf_to_pdbs(self.vars, sdf_dir, sdf_dir)
        pdb_dir = sdf_dir + "_PDB"
        if not os.path.exists(pdb_dir):
            raise RuntimeError(f"PDB目录未生成: {pdb_dir}")

        # 只保留每个SMILES的第一个PDB
        keep_one_pdb_per_smiles(pdb_dir)

        # 3. PDB转PDBQT（加进度条）
        file_conversion_class = self.pick_conversion_class(self.vars["conversion_choice"])
        file_converter = file_conversion_class(
            self.vars,
            self.vars["filename_of_receptor"],
            test_boot=False
        )
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        pdbqt_files = []
        for pdb_file in tqdm(pdb_files, desc="PDB转PDBQT进度"):
            try:
                pdbqt_file = pdb_file + "qt"
                if not os.path.exists(pdbqt_file):
                    cmd = [
                        self.vars["mgl_python"],
                        self.vars["prepare_ligand4.py"],
                        "-l", pdb_file,
                        "-o", pdbqt_file
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(pdbqt_file):
                    pdbqt_files.append(pdbqt_file)
                time.sleep(0.5)
            except Exception as e:
                continue
        if not pdbqt_files:
            raise RuntimeError("没有成功生成任何PDBQT文件,无法进行后续对接。请检查MGLTools配置和PDB文件内容。")
        return pdb_dir  # 返回PDBQT文件所在目录

    def run_docking(self, receptor_pdbqt: str, ligand_dir: str) -> str:
        print("执行分子对接...")

        results_dir = os.path.join(self.vars["output_directory"], "docking_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        ligand_files = glob.glob(os.path.join(ligand_dir, "*.pdbqt"))
        if not ligand_files:
            raise RuntimeError("没有找到任何PDBQT文件,无法进行对接。")

        max_workers = min(8, os.cpu_count() or 1)
        futures = []
        scores = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ligand_file in ligand_files:
                futures.append(executor.submit(
                    vina_dock_single, ligand_file, receptor_pdbqt, results_dir, self.vars
                ))
            for future in tqdm(as_completed(futures), total=len(futures), desc="对接进度"):
                ligand_file, success, score = future.result()
                mol_name = os.path.basename(ligand_file).replace(".pdbqt", "")
                scores[mol_name] = score

        # 输出smi+score文件
        output_smiles_scores(self.vars["ligands"], scores, os.path.join(results_dir, "final_scored.smi"))
        return results_dir
        
    @staticmethod
    def pick_conversion_class(conversion_choice: str) -> type:
        """
        选择文件转换类
        
        参数:
        :param str conversion_choice: 转换方式选择
        
        返回:
        :returns: type: 转换类
        """
        conversion_classes = {
            "MGLToolsConversion": MGLToolsConversion,
            "ObabelConversion": ObabelConversion
        }
        return conversion_classes.get(conversion_choice)
        
    @staticmethod
    def pick_docking_class(dock_choice: str) -> type:
        """
        选择对接类
        
        参数:
        :param str dock_choice: 对接程序选择
        
        返回:
        :returns: type: 对接类
        """
        docking_classes = {
            "VinaDocking": VinaDocking,
            "QuickVina2Docking": QuickVina2Docking
        }
        return docking_classes.get(dock_choice)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分子对接工作流程")
    
    # 必需参数
    parser.add_argument("--receptor", default="tutorial/PARP/4r6eA_PARP1_prepared.pdb")
    parser.add_argument("--ligands", default="datasets/source_compounds/naphthalene_smiles.smi")
    parser.add_argument("--output", default="test_docking_finetune_523")
    
    # 可选参数
    parser.add_argument("--dock_choice", default="VinaDocking",
                      choices=["VinaDocking", "QuickVina2Docking"])
    parser.add_argument("--conversion_choice", default="MGLToolsConversion",
                      choices=["MGLToolsConversion", "ObabelConversion"])
    parser.add_argument("--center_x", type=float, default=-70.76)
    parser.add_argument("--center_y", type=float, default=21.82)
    parser.add_argument("--center_z", type=float, default=28.33)
    parser.add_argument("--size_x", type=float, default=25.0)
    parser.add_argument("--size_y", type=float, default=16.0)
    parser.add_argument("--size_z", type=float, default=25.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=9)
    parser.add_argument("--num_processors", type=int)
    parser.add_argument("--max_variants_per_compound", type=int, default=3)
    #gypsum 
    parser.add_argument("--gypsum_thoroughness", type=int, default=3)
    parser.add_argument("--gypsum_timeout_limit", type=int, default=15)
    parser.add_argument("--min_ph", type=float, default=6.4)
    parser.add_argument("--max_ph", type=float, default=8.4)
    parser.add_argument("--pka_precision", type=float, default=1.0)
    #output
    # 添加文件转换相关参数
    parser.add_argument("--sdf_dir", default="ligands3D_SDFs")
    parser.add_argument("--pdb_dir", default="ligands3D_PDBs")
    
    


    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 构建配置字典
    config = {
        "filename_of_receptor": args.receptor,
        "dock_choice": args.dock_choice,
        "conversion_choice": args.conversion_choice,
        "output_directory": args.output,
        "center_x": args.center_x,
        "center_y": args.center_y,
        "center_z": args.center_z,
        "size_x": args.size_x,
        "size_y": args.size_y,
        "size_z": args.size_z,
        "docking_exhaustiveness": args.exhaustiveness,
        "docking_num_modes": args.num_modes,
        "number_of_processors": args.num_processors,
        "debug_mode": False,
        "max_variants_per_compound": args.max_variants_per_compound,
        "gypsum_thoroughness": args.gypsum_thoroughness,
        "gypsum_timeout_limit": args.gypsum_timeout_limit,
        "min_ph": args.min_ph,
        "max_ph": args.max_ph,
        "pka_precision": args.pka_precision,
        #output
        "sdf_dir": os.path.join(args.output, "ligands3D_SDFs"),
        "pdb_dir": os.path.join(args.output, "ligands3D_PDBs"),
        "ligand_dir": os.path.join(args.output, "ligands"),
        "ligands": args.ligands,
        
        
        # 添加parallelizer
        "parallelizer": Parallelizer(mode="serial", num_procs=1),
        # MGLTools相关路径
        "mgl_python": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/bin/pythonsh"),
        "prepare_receptor4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
        "prepare_ligand4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"),
        # 添加Vina执行文件路径
        "docking_executable": os.path.join(PROJECT_ROOT, "autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina")
    }
    
    try:
        # 创建工作流程实例
        workflow = DockingWorkflow(config)
        
        # 准备受体
        receptor_pdbqt = workflow.prepare_receptor()
        
        # 准备配体
        ligand_dir = workflow.prepare_ligands(args.ligands)
        
        # 执行对接
        results_dir = workflow.run_docking(receptor_pdbqt, ligand_dir)
        
        print(f"对接完成！结果保存在: {results_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()



