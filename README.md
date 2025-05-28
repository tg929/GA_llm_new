##########524执行命令##########
conda activate fraggpt
####GA_llm_finetune.py
1.
python GA_llm_finetune.py --output_dir output_524_filtest0 --LipinskiStrictFilter
2.
python GA_llm_finetune.py --output_dir output_524_filtest0 --LipinskiStrictFilter --top_mols_to_seed_next_generation 50 --diversity_mols_to_seed_first_generation 50 --diversity_seed_depreciation_per_gen 10

####GA_llm_modified_first.py
1.
python GA_llm_modified_first.py --output_dir output_524_firstest --LipinskiStrictFilter
2.
python GA_llm_modified_first.py --output_dir output_524_firstest_0 --LipinskiStrictFilter --top_mols_to_seed_next_generation 50 --diversity_mols_to_seed_first_generation 50 --diversity_seed_depreciation_per_gen 10 

####GA_llm_modified_second.py
1.
python GA_llm_modified_second.py --output_dir output_524_secondest --LipinskiStrictFilter
2.
python GA_llm_modified_second.py --output_dir output_524_secondest_0 --LipinskiStrictFilter --top_mols_to_seed_next_generation 50 --diversity_mols_to_seed_first_generation 50 --diversity_seed_depreciation_per_gen 10


526执行
#######GA_llm_rga.py
python GA_llm_rga.py --output_dir output_rga --top_mols_to_seed_next_generation 50 --diversity_mols_to_seed_first_generation 50 --diversity_seed_depreciation_per_gen 10 --LipinskiStrictFilter --parallel 

528执行
#####GA_llm_rga.py
python GA_llm_rga.py  --output_dir output_rga_528 --parallel --top_mols_to_seed_next_generati
on 50 --diversity_mols_to_seed_first_generation 50 --diversity_seed_depreciation_per_gen 10 --LipinskiStrictFilter

#多目标
test_multi_objective.py
###多目标执行文件
#####GA_llm_rga.py
python GA_llm_rga.py --output_dir output_528 --parallel --LipinskiStrictFilter
多目标执行：
python GA_llm_rga.py \
    --output_dir output_rga_528_mutli \
    --top_mols_to_seed_next_generation 10 \
    --diversity_mols_to_seed_first_generation 10 \
    --diversity_seed_depreciation_per_gen 2 \
    --LipinskiStrictFilter \
    --parallel \
    --use_multi_objective
单目标执行
python GA_llm_rga.py \
    --output_dir output_rga_528_single \
    --top_mols_to_seed_next_generation 10 \
    --diversity_mols_to_seed_first_generation 10 \
    --diversity_seed_depreciation_per_gen 2 \
    --LipinskiStrictFilter \
    --parallel \
    --use_single_objective