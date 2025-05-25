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
