#!/bin/bash
cd /ocean/projects/cis260044p/shared/src/visualwebarena/

python run.py \
   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
   --test_start_idx 0   --test_end_idx 10  \
   --result_dir ./results/  \
   --test_config_base_dir=config_files/vwa/test_classifieds  \
   --model Qwen/Qwen2.5-VL-3B-Instruct   \
   --action_set_tag som  --observation_type image_som \
   --provider openai --max_obs_length 15360 --mode chat
