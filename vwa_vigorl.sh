#!/bin/bash

python run.py \
   --instruction_path agent/prompts/jsons/p_vigorl_som_cot_id_actree_3s.json \
   --test_start_idx 0   --test_end_idx 1  \
   --result_dir ./results_test_vigorl/  \
   --test_config_base_dir=config_files/vwa/test_reddit  \
   --model gsarch/ViGoRL-Multiturn-7b-Web-Action   \
   --action_set_tag som  --observation_type image_som \
   --provider vigorl --max_obs_length 15360 --mode chat \
   --agent_type vigorl
