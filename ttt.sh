# MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_NAME="${MODEL##*/}"




# Full random experiment :
nohup python run/run_alfworld.py \
  --config configs/rl_alf_config.yaml \
  --set llm.model=$MODEL \
  --set llm.base_url=http://10.225.68.29:1700/v1 \
  --set experiment.num_sections=5 \
  --set memory.k_retrieve=5 \
  --set memory.retrieve_strategy=random_full \
  --set rl_config.topk=5 \
  --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_random_full \
  > ./log/alfworld_${MODEL_NAME}_topk5_random_full.log 2>&1 &

  # - Epoch 0: in-dist 42.14% (59/140), out-dist 50.00% (67/134)
  # - Epoch 1: in-dist 30.00% (42/140), out-dist 45.52% (61/134)
  # - Epoch 2: in-dist 37.86% (53/140), out-dist 50.00% (67/134)
  # - Epoch 3: in-dist 37.14% (52/140), out-dist 46.27% (62/134)


# Random partial experiment : 
nohup python run/run_alfworld.py \
  --config configs/rl_alf_config.yaml \
  --set llm.model=$MODEL \
  --set llm.base_url=http://10.225.68.29:1701/v1 \
  --set experiment.num_sections=5 \
  --set memory.k_retrieve=5 \
  --set memory.retrieve_strategy=random_partial \
  --set rl_config.topk=5 \
  --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_random_partial \
  > ./log/alfworld_${MODEL_NAME}_topk5_random_partial.log 2>&1 &
  # - Epoch 0: in-dist 44.29% (62/140), out-dist 46.27% (62/134)
  # - Epoch 1: in-dist 45.71% (64/140), out-dist 46.27% (62/134)
  # - Epoch 2: in-dist 45.71% (64/140), out-dist 49.25% (66/134)
  # - Epoch 3: in-dist 47.14% (66/140), out-dist 46.27% (62/134)
  # - Epoch 4: in-dist 50.71% (71/140), out-dist 41.79% (56/134)










################################ thompson sampling ##################### 

nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.24:1700/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=5 \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k5_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k5_topk3.log 2>&1 &
  # - Epoch 0: in-dist 41.43% (58/140), out-dist 43.28% (58/134)
  # - Epoch 1: in-dist 64.29% (90/140), out-dist 68.66% (92/134)
  # - Epoch 2: in-dist 68.57% (96/140), out-dist 68.66% (92/134)
  # - Epoch 3: in-dist 66.43% (93/140), out-dist 64.18% (86/134)

nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.24:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k10_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k10_topk3.log 2>&1 &

  # - Epoch 0: in-dist 43.57% (61/140), out-dist 47.76% (64/134)
  # - Epoch 1: in-dist 67.14% (94/140), out-dist 69.40% (93/134)
  # - Epoch 2: in-dist 60.71% (85/140), out-dist 73.88% (99/134)
  # - Epoch 3: in-dist 66.43% (93/140), out-dist 75.37% (101/134)


nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.24:1700/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=20 \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k20_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k20_topk3.log 2>&1 &


nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.24:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=30 \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k30_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k30_topk3.log 2>&1 &



nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1700/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=5 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k10_topk5 \
    > ./log/alfworld_ts_${MODEL_NAME}_k10_topk5.log 2>&1 &


nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=7 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k10_topk7 \
    > ./log/alfworld_ts_${MODEL_NAME}_k10_topk7.log 2>&1 &



nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.24:1700/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=10 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k10_topk10 \
    > ./log/alfworld_ts_${MODEL_NAME}_k10_topk10.log 2>&1 &


############################ turn off ts ###########
nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1700/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=5 \
    --set rl_config.use_thompson_sampling=false \
    --set experiment.experiment_name=alfworld_no_ts_${MODEL_NAME}_k10_topk5 \
    > ./log/alfworld_no_ts_${MODEL_NAME}_k10_topk5.log 2>&1 &


nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=false \
    --set experiment.experiment_name=alfworld_no_ts_${MODEL_NAME}_k10_topk3 \
    > ./log/alfworld_no_ts_${MODEL_NAME}_k10_topk3.log 2>&1 &



############################ Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 ###########

nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --set llm.base_url=http://10.225.68.16:1700/v1 \
    --set experiment.num_sections=10 \
    --set memory.k_retrieve=10 \
    --set rl_config.topk=4 \
    --set rl_config.use_thompson_sampling=false \
    --set experiment.experiment_name=alfworld_no_ts_Qwen3-30B-A3B-Instruct-2507-FP8_k10_topk4 \
    > ./log/alfworld_no_ts_Qwen3-30B-A3B-Instruct-2507-FP8_k10_topk4.log 2>&1 &



nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --set llm.base_url=http://10.225.68.16:1701/v1 \
    --set experiment.num_sections=10 \
    --set memory.k_retrieve=4 \
    --set rl_config.topk=4 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_Qwen3-30B-A3B-Instruct-2507-FP8_k4_topk4 \
    > ./log/alfworld_ts_Qwen3-30B-A3B-Instruct-2507-FP8_k4_topk4.log 2>&1 &
















############################################################

pkill -f -9 run_alfworld.py

codex resume 019d4453-c409-7b53-95d8-b47945d5bcb0

cat alfworld_Qwen2.5-7B-Instruct_topk5_random_full.log|grep Epoch|grep complete