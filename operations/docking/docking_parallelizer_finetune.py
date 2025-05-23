import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from docking_demo_finetune import DockingWorkflow
from autogrow.operators.convert_files.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer

def process_ligand_wrapper(workflow, smile):
    """包装器函数，用于并行处理单个配体"""
    try:
        # 为每个分子创建临时目录
        temp_dir = os.path.join(workflow.vars["output_directory"], "temp", f"ligand_{hash(smile)}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 准备配体文件
        ligand_dir = workflow.prepare_ligands(smile)
        
        # 执行对接
        results_dir = workflow.run_docking(workflow.receptor_pdbqt, ligand_dir)
        
        # 清理临时文件
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            
        return {
            'smile': smile,
            'docking_score': workflow.get_last_score()
        }
    except Exception as e:
        logging.error(f"处理配体时发生错误 {smile}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='并行分子对接脚本')
    parser.add_argument('--receptor', default="tutorial/PARP/4r6eA_PARP1_prepared.pdb", help='受体PDB文件路径')
    parser.add_argument('--output', default="test_docking_parallel", help='输出目录')
    parser.add_argument('--smiles', default="datasets/source_compounds/naphthalene_smiles.smi", help='包含SMILES字符串的文件路径')
    parser.add_argument('--num_processes', type=int, default=mp.cpu_count(), 
                      help='并行进程数(默认使用所有可用CPU核心)')
    args = parser.parse_args()

    # 设置日志
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output, "docking_parallel.log")),
            logging.StreamHandler()
        ]
    )

    # 构建配置字典
    config = {
        "filename_of_receptor": args.receptor,
        "dock_choice": "VinaDocking",
        "conversion_choice": "MGLToolsConversion",
        "output_directory": args.output,
        "center_x": -70.76,
        "center_y": 21.82,
        "center_z": 28.33,
        "size_x": 25.0,
        "size_y": 16.0,
        "size_z": 25.0,
        "docking_exhaustiveness": 8,
        "docking_num_modes": 9,
        "number_of_processors": args.num_processes,
        "debug_mode": False,
        "max_variants_per_compound": 3,
        "gypsum_thoroughness": 3,
        "gypsum_timeout_limit": 15,
        "min_ph": 6.4,
        "max_ph": 8.4,
        "pka_precision": 1.0,
        "sdf_dir": os.path.join(args.output, "ligands3D_SDFs"),
        "pdb_dir": os.path.join(args.output, "ligands3D_PDBs"),
        "ligand_dir": os.path.join(args.output, "ligands"),
        "ligands": args.smiles,
        "parallelizer": Parallelizer(mode="multiprocessing", num_procs=args.num_processes),
        "mgl_python": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/bin/pythonsh"),
        "prepare_receptor4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
        "prepare_ligand4.py": os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"),
        "docking_executable": os.path.join(PROJECT_ROOT, "autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina")
    }

    # 初始化工作流程
    workflow = DockingWorkflow(config)
    
    # 准备受体（只需要做一次）
    workflow.receptor_pdbqt = workflow.prepare_receptor()

    # 读取SMILES
    with open(args.smiles, 'r') as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]

    # 创建进程池
    pool = mp.Pool(processes=args.num_processes)
    
    # 创建偏函数，固定workflow参数
    process_func = partial(process_ligand_wrapper, workflow)

    # 使用tqdm显示进度
    results = []
    for result in tqdm(pool.imap(process_func, smiles_list), 
                      total=len(smiles_list),
                      desc="并行处理配体"):
        if result:
            results.append(result)

    # 关闭进程池
    pool.close()
    pool.join()

    # 保存结果
    with open(os.path.join(args.output, 'docking_results_parallel.smi'), 'w') as f:
        f.write("smiles\tscore\n")
        for result in results:
            f.write(f"{result['smile']}\t{result['docking_score']}\n")

    logging.info(f"处理完成。成功处理 {len(results)}/{len(smiles_list)} 个配体")

if __name__ == "__main__":
    main()
