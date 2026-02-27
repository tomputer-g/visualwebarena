

## 1. GCP and Vertex

- Create a Google Cloud project (or use an existing one).
- In Cloud Console enable the **Vertex AI API** for that project.
- Turn on **billing** for the project (Vertex needs it; you can stay in free tier and pay nothing).
- On the machine where you will run evals, install the gcloud CLI and run:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

Use your real project ID. After this, Python can use Vertex without an API key.



## 2. Start the sites on EC2

SSH into the EC2 box and start the VWA sites (Docker + homepage). Example:

```bash
ssh -i /path/to/key.pem ubuntu@YOUR_EC2_PUBLIC_DNS
cd /home/ubuntu
./vwa_launch.sh
```

Leave that running (or run the last command in tmux/screen). To reset sites between runs: `./vwa_reset.sh`(or whatever you named the script) on the EC2.

## 3. Run evals on your machine

Use a terminal on your laptop. Activate the venv and go to the repo root.

Set env vars so the benchmark talks to your EC2 and GCP project:

```bash
export DATASET=visualwebarena
export CLASSIFIEDS="http://YOUR_EC2_DNS:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://YOUR_EC2_DNS:7770"
export REDDIT="http://YOUR_EC2_DNS:9999"
export WIKIPEDIA="http://YOUR_EC2_DNS:8888"
export HOMEPAGE="http://YOUR_EC2_DNS:4399"
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

Replace `YOUR_EC2_DNS` and `YOUR_PROJECT_ID`. If you use a different classifieds reset token on EC2, set `CLASSIFIEDS_RESET_TOKEN` to that value.

Generate task configs and login cookies once:

```bash
python scripts/generate_test_data.py
bash prepare.sh
```

Run one Classifieds task (task 0):

```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 --test_end_idx 1 \
  --result_dir results_gemini_flash \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --provider google --model gemini --mode completion \
  --max_obs_length 15360 --action_set_tag som --observation_type image_som
```

Results go under the dir you pass to `--result_dir` (e.g. `results_gemini_flash/`). Each task gets a `render_<id>.html` and the log prints PASS or FAIL.

To run more tasks, change the range and optionally the result dir:

```bash
# First 10 Classifieds tasks
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 --test_end_idx 10 \
  --result_dir results_gemini_classifieds_0_10 \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --provider google --model gemini --mode completion \
  --max_obs_length 15360 --action_set_tag som --observation_type image_som
```

For Shopping use `--test_config_base_dir config_files/vwa/test_shopping`. For Reddit use `config_files/vwa/test_reddit`. Shopping tasks need the cookie file from `prepare.sh` (e.g. `.auth/shopping_state.json`); if it is missing, the run will error for those tasks.

