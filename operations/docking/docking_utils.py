# Code adapted from https://github.com/SeulLee05/MOOD/blob/main/scorer/docking.py
#其实没有用到
import sys
import os
from shutil import rmtree
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel
import tempfile, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
class DockingVina(object):
    def __init__(self, target):
        super().__init__()
        if target == '4r6e':
            self.box_center = ( -70.76,21.82,28.33)
            self.box_size = (15.0,15.0,15.0)
        elif target == '3pbl':
            self.box_center = (9, 22.5, 26)
            self.box_size = (15.0,15.0,15.0)
        elif target == '1iep':
            self.box_center = (15.6138918, 53.38013513, 15.454837)
            self.box_size = (15.0,15.0,15.0)
        elif target == '2rgp':
            self.box_center = (16.29212, 34.870818, 92.0353)
            self.box_size = (15.0,15.0,15.0)
        elif target == '3eml':
            self.box_center = (-9.06363, -7.1446, 55.86259999)
            self.box_size = (15.0,15.0,15.0)
        elif target == '3ny8':
            self.box_center = (2.2488, 4.68495, 51.39820000000001)
            self.box_size = (15.0,15.0,15.0)   
        elif target == '4rlu':
            self.box_center = (-0.73599, 22.75547, -31.23689)
            self.box_size = (15.0,15.0,15.0)
        elif target == '4unn':
            self.box_center = (5.684346153, 18.1917, -7.3715)
            self.box_size = (15.0,15.0,15.0)
        elif target == '5mo4':
            self.box_center = (-44.901, 20.490354, 8.48335)
            self.box_size = (15.0,15.0,15.0)
        elif target == '7l11':
            self.box_center = (-21.81481, -4.21606, -27.98378)
            self.box_size = (15.0,15.0,15.0)            
        self.protein = target
        self.vina_program = f'autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina'
        self.receptor_file = f'pdb/{target}.pdbqt'
        self.exhaustiveness = 1  #从是随机配体结构开始的独立运行的数量
        #(每一次运行都由连续的局部优化步骤组成，其中包括对评分函数及其在位置-方向-扭矩坐标中的导数的许多评估)。
        #Exhaustiveness通常直接与运行时间相关。Exhaustiveness越低对接速度越快，Exhaustiveness越高搜索空间更全面
        self.num_sub_proc = 1#
        self.num_cpu_dock = 32#对接CPU
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100
        tmp_base = os.path.join(PROJECT_ROOT, "docking/tmp/")
        os.makedirs(tmp_base, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="tmp_", dir=tmp_base)
        print(f'Docking tmp dir: {tmp_dir}')
        self.temp_dir = tmp_dir
        # i = 0
        # while True:
        #     tmp_dir = f'./utils/docking/tmp/tmp{i}'
        #     if not os.path.exists(tmp_dir):
        #         print(f'Docking tmp dir: {tmp_dir}')
        #         os.makedirs(tmp_dir)
        #         self.temp_dir = tmp_dir
        #         break
        #     i += 1

    def gen_3d(self, smi, ligand_mol_file):#obabel生成3d结构
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):#vina对接
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_mol_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        #pybel:openbabel的python接口
        ms = list(pybel.readfile("mol", ligand_mol_file))#配体
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        #run_line    vina_program 调用
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        #subprocess.check_output:运行命令并返回输出
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):#
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print(e)
                print("gen_3d unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 1.0
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 1.0
                continue
            if len(affinity_list) == 0:
                affinity_list.append(1.0)

            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        """
            input SMILES list
            output affinity list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed')