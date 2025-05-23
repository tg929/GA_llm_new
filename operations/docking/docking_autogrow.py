import os
import sys
import argparse
import logging
import shutil
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
# --- AutoGrow模块导入 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0, PROJECT_ROOT)

from autogrow.operators.convert_files.conversion_to_3d import convert_to_3d
from autogrow.operators.convert_files.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion
# from autogrow.docking.docking_class.docking_file_conversion.convert_with_obabel import ObabelConversion # 备选
from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# --- 1. 参数解析与配置 ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="SMILES to Docking Score using AutoGrow4.0 modules.")
    # 输入/输出
    parser.add_argument("--smi_file",  default="datasets/source_compounds/naphthalene_smiles.smi")
    parser.add_argument("--receptor_pdb", default="tutorial/PARP/4r6eA_PARP1_prepared.pdbqt")
    parser.add_argument("--output_smi", default="docking_autogrow_output/naphthalene_smiles_autogrow_docking_scores.smi")
    # 路径配置
    parser.add_argument("--mgltools_path", default=os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/bin/pythonsh"))
    parser.add_argument("--vina_executable", default=os.path.join(PROJECT_ROOT, "autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina")) # 直接指向Vina可执行文件
    parser.add_argument("--output_dir", default="docking_autogrow_output", help="Directory for all outputs and temporary files.")
    # 对接盒子参数
    parser.add_argument("--center_x", type=float, default=-70.76)
    parser.add_argument("--center_y", type=float, default=21.82)
    parser.add_argument("--center_z", type=float, default=28.33)
    parser.add_argument("--size_x", type=float, default=25.0)
    parser.add_argument("--size_y", type=float, default=16.0)
    parser.add_argument("--size_z", type=float, default=25.0)

    # Gypsum-DL 参数
    parser.add_argument("--max_variants_per_compound", type=int, default=1, help="Gypsum: max variants per compound.")
    parser.add_argument("--gypsum_thoroughness", type=int, default=3, help="Gypsum: thoroughness level.")
    parser.add_argument("--gypsum_timeout_limit", type=int, default=15, help="Gypsum: timeout per SMILES (seconds).")
    parser.add_argument("--min_ph", type=float, default=6.4)
    parser.add_argument("--max_ph", type=float, default=8.4)
    parser.add_argument("--pka_precision", type=float, default=1.0)
    # Vina 参数
    parser.add_argument("--docking_exhaustiveness", type=int, default=8, help="Vina: exhaustiveness.")
    # 并行处理
    parser.add_argument("--num_cpus", type=int, default=max(1, os.cpu_count() - 1), help="Number of CPUs for parallel processing.")
    return parser.parse_args()

def setup_vars(args):
    vars_config = {
        "root_output_folder": os.path.abspath(args.output_dir),
        "mgl_python": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/bin/pythonsh"),
        "prepare_ligand4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"),
        "prepare_receptor4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
        "docking_executable": os.path.join(PROJECT_ROOT, "autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina"),
        "center_x": args.center_x, "center_y": args.center_y, "center_z": args.center_z,
        "size_x": args.size_x, "size_y": args.size_y, "size_z": args.size_z,
        "docking_exhaustiveness": args.docking_exhaustiveness,
        "docking_num_modes": 9,
        "parallelizer": Parallelizer(mode="serial", num_procs=1),
        "max_variants_per_compound": args.max_variants_per_compound,
        "gypsum_thoroughness": args.gypsum_thoroughness,
        "gypsum_timeout_limit": args.gypsum_timeout_limit,
        "min_ph": args.min_ph, "max_ph": args.max_ph, "pka_precision": args.pka_precision,
        "debug_mode": False
    }
    return vars_config

# --- 2. 受体准备 ---
def prepare_receptor(vars_config, receptor_pdb):
    receptor_pdbqt = receptor_pdb.replace(".pdb", ".pdbqt")
    if receptor_pdb.endswith(".pdbqt"):
        return receptor_pdb
    if os.path.exists(receptor_pdbqt) and os.path.getsize(receptor_pdbqt) > 0:
        return receptor_pdbqt
    converter = MGLToolsConversion(vars_manager=vars_config, receptor_file=receptor_pdb)
    converter.convert_receptor_pdb_files_to_pdbqt(
        receptor_pdb, vars_config["mgl_python"], vars_config["prepare_receptor4.py"], 1
    )
    if not os.path.exists(receptor_pdbqt):
        raise RuntimeError(f"受体PDBQT生成失败: {receptor_pdbqt}")
    return receptor_pdbqt


# --- 3. 配体处理工作函数 (用于并行化) ---   重要
def process_single_smiles(smiles, mol_id, vars_config, receptor_pdbqt, output_dir):
    temp_dir = os.path.join(output_dir, "temp", f"{mol_id}_{os.urandom(4).hex()}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        # 1. SMILES→3D SDF
        temp_smi = os.path.join(temp_dir, f"{mol_id}.smi")
        with open(temp_smi, "w") as f:
            f.write(f"{smiles}\t{mol_id}\n")
        convert_to_3d(vars_config, temp_smi, temp_dir)
        sdf_dir = os.path.join(temp_dir, "3D_SDFs")
        sdf_files = glob.glob(os.path.join(sdf_dir, f"{mol_id}*.sdf"))
        if not sdf_files:
            return (smiles, mol_id, "NA")
        sdf_file = sdf_files[0]
        # 2. SDF→PDBQT
        lig_converter = MGLToolsConversion(vars_manager=vars_config, receptor_file=receptor_pdbqt)
        pdbqt_file = lig_converter.convert_ligand_sdf_to_pdbqt(sdf_file)
        if not pdbqt_file or not os.path.exists(pdbqt_file):
            return (smiles, mol_id, "NA")
        # 3. 对接
        docker = VinaDocking(vars_manager=vars_config, receptor_file=receptor_pdbqt)
        score = docker.run_dock(file_path_to_ligand_pdbqt=pdbqt_file)
        return (smiles, mol_id, str(score) if score is not None else "NA")
    except Exception as e:
        return (smiles, mol_id, "NA")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- 4. 主逻辑 ---
def main():
    args = parse_arguments()
    
    # 设置日志
    log_dir = os.path.join(args.output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, # DEBUG, INFO, WARNING, ERROR, CRITICAL
        format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "docking_pipeline.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"脚本启动，参数: {args}")

    
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # project_root_for_autogrow = os.path.dirname(os.path.dirname(current_script_path)) # 假设脚本在 project_root/custom_scripts/script.py
    project_root_for_autogrow = PROJECT_ROOT # 使用顶部定义的 PROJECT_ROOT

    vars_config = setup_vars(args)
    
    # 准备受体 (只需一次)
    try:
        abs_receptor_pdb = os.path.abspath(args.receptor_pdb)
        if not os.path.exists(abs_receptor_pdb):
            logging.error(f"受体PDB文件未找到: {abs_receptor_pdb}")
            sys.exit(1)
        prepared_receptor_pdbqt = prepare_receptor(vars_config, abs_receptor_pdb)
    except Exception as e:
        logging.error(f"受体准备失败，程序终止: {e}", exc_info=True)
        sys.exit(1)

    # 读取SMILES文件
    smiles_to_process = []
    try:
        with open(args.smi_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split() # 假定SMILES和ID通过空格或制表符分隔
                if len(parts) >= 2:
                    smiles_to_process.append((parts[0], parts[1])) # (SMILES, ID)
                elif len(parts) == 1:
                    smiles_to_process.append((parts[0], f"mol_{i+1}")) # 如果没有ID，自动生成
                else:
                    logging.warning(f"跳过格式不正确的行: {line}")
        logging.info(f"从 {args.smi_file} 读取到 {len(smiles_to_process)} 个SMILES分子。")
    except Exception as e:
        logging.error(f"读取SMILES文件 {args.smi_file} 失败: {e}", exc_info=True)
        sys.exit(1)

    if not smiles_to_process:
        logging.info("没有SMILES分子需要处理。")
        sys.exit(0)
        
    # 并行处理所有SMILES
    all_results = []
    with ProcessPoolExecutor(max_workers=args.num_cpus) as executor:
        futures = [
            executor.submit(process_single_smiles, smi, mol_id, vars_config, prepared_receptor_pdbqt, args.output_dir)
            for smi, mol_id in smiles_to_process
        ]
        for future in tqdm(as_completed(futures), total=len(smiles_to_process), desc="处理SMILES分子"):
            try:
                result_tuple = future.result() # (original_smi, original_id, docking_score)
                all_results.append(result_tuple)
            except Exception as e_future:
                logging.error(f"一个并行任务失败: {e_future}", exc_info=True)
                # 可以考虑如何处理失败的任务，例如记录SMILES和ID
                # all_results.append((smi_tuple_from_somewhere, "NA_FutureError")) 

    # --- 5. 输出结果 ---
    output_smi_file_abs = os.path.abspath(args.output_smi)
    os.makedirs(os.path.dirname(output_smi_file_abs), exist_ok=True) #确保输出目录存在
    
    # 根据ID对结果进行排序（可选，但能让输出与输入顺序更一致，如果输入有特定顺序）
    # all_results.sort(key=lambda x: x[1]) # 按ID排序

    with open(output_smi_file_abs, "w") as f_out:
        f_out.write("SMILES\tScore\tID\n") # 添加表头
        for smi_str, smi_id, score in all_results:
            f_out.write(f"{smi_str}\t{score}\t{smi_id}\n")
            
    logging.info(f"处理完成。结果已写入: {output_smi_file_abs}")
    num_successful = sum(1 for _, _, score in all_results if score not in ["NA", "NA_Error", "NA_FutureError"])
    logging.info(f"成功获取分数的分子数量: {num_successful} / {len(all_results)}")

if __name__ == "__main__":
    main()