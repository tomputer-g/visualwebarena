#!/bin/bash

python run.py \
   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
   --test_start_idx 341   --test_end_idx 1000  \
   --result_dir ./results_all_shopping_actree_qwen_baseline_mar23/  \
   --test_config_base_dir=config_files/vwa/test_shopping  \
   --model Qwen/Qwen2.5-VL-7B-Instruct   \
   --action_set_tag som  --observation_type image_som \
   --provider openai --max_obs_length 15360 --mode chat
