# MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_NAME="${MODEL##*/}"



# disable tri channel, topk 5 positive
nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=$MODEL \
    --set llm.base_url=http://10.225.68.29:1700/v1 \
    --set experiment.num_sections=5 \
    --set memory.k_retrieve=5 \
    --set rl_config.tri_channel_enabled=false \
    --set rl_config.topk=5 \
    --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_trichannel_off \
    > ./log/alfworld_${MODEL_NAME}_topk5_trichannel_off.log 2>&1 &

  # - Epoch 0: in-dist 44.29% (62/140), out-dist 48.51% (65/134)
  # - Epoch 1: in-dist 51.43% (72/140), out-dist 52.99% (71/134)
  # - Epoch 2: in-dist 47.14% (66/140), out-dist 46.27% (62/134)
  # - Epoch 3: in-dist 57.86% (81/140), out-dist 64.93% (87/134)
  # - Epoch 4: in-dist 59.29% (83/140), out-dist 64.93% (87/134)
  # - Epoch 5: in-dist 61.43% (86/140), out-dist 63.43% (85/134)


# disable tri channel, topk 3 positive
nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=$MODEL \
    --set llm.base_url=http://10.225.68.24:1700/v1 \
    --set experiment.num_sections=5 \
    --set memory.k_retrieve=5 \
    --set rl_config.tri_channel_enabled=false \
    --set rl_config.topk=3 \
    --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk3_trichannel_off \
    > ./log/alfworld_${MODEL_NAME}_topk3_trichannel_off.log 2>&1 &
  # - Epoch 0: in-dist 42.14% (59/140), out-dist 42.54% (57/134)
  # - Epoch 1: in-dist 47.86% (67/140), out-dist 50.75% (68/134)
  # - Epoch 2: in-dist 47.14% (66/140), out-dist 49.25% (66/134)
  # - Epoch 3: in-dist 61.43% (86/140), out-dist 69.92% (93/133)
  # - Epoch 4: in-dist 57.14% (80/140), out-dist 63.43% (85/134)




# Full random experiment :
nohup python run/run_alfworld.py \
  --config configs/rl_alf_config.yaml \
  --set llm.model=$MODEL \
  --set llm.base_url=http://10.225.68.29:1700/v1 \
  --set experiment.num_sections=5 \
  --set memory.k_retrieve=5 \
  --set memory.retrieve_strategy=random_full \
  --set rl_config.tri_channel_enabled=false \
  --set rl_config.topk=5 \
  --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_random_full \
  > ./log/alfworld_${MODEL_NAME}_topk5_random_full.log 2>&1 &

# Random partial experiment : 
nohup python run/run_alfworld.py \
  --config configs/rl_alf_config.yaml \
  --set llm.model=$MODEL \
  --set llm.base_url=http://10.225.68.29:1701/v1 \
  --set experiment.num_sections=5 \
  --set memory.k_retrieve=5 \
  --set memory.retrieve_strategy=random_partial \
  --set rl_config.tri_channel_enabled=false \
  --set rl_config.topk=5 \
  --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_random_partial \
  > ./log/alfworld_${MODEL_NAME}_topk5_random_partial.log 2>&1 &




# enable tri channel, topk 5: 3+1+1
nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=$MODEL \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=5 \
    --set memory.k_retrieve=5 \
    --set rl_config.tri_channel_enabled=true \
    --set rl_config.k_pos=3 \
    --set rl_config.k_neg=1 \
    --set rl_config.k_zero=1 \
    --set rl_config.q_epsilon=0.05 \
    --set rl_config.uncertain_visit_threshold=2 \
    --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk5_trichannel_on \
    > ./log/alfworld_${MODEL_NAME}_topk5_trichannel_on_311.log 2>&1 &
  # - Epoch 0: in-dist 46.43% (65/140), out-dist 50.00% (67/134)
  # - Epoch 1: in-dist 40.00% (56/140), out-dist 38.81% (52/134)
  # - Epoch 2: in-dist 36.43% (51/140), out-dist 36.57% (49/134)
  # - Epoch 3: in-dist 57.86% (81/140), out-dist 64.18% (86/134)
  # - Epoch 4: in-dist 59.29% (83/140), out-dist 53.73% (72/134)
  # - Epoch 5: in-dist 57.14% (80/140), out-dist 58.96% (79/134)


# enable tri channel, topk 3: 1+1+1
nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=$MODEL \
    --set llm.base_url=http://10.225.68.24:1701/v1 \
    --set experiment.num_sections=5 \
    --set memory.k_retrieve=5 \
    --set rl_config.tri_channel_enabled=true \
    --set rl_config.k_pos=1 \
    --set rl_config.k_neg=1 \
    --set rl_config.k_zero=1 \
    --set rl_config.q_epsilon=0.05 \
    --set rl_config.uncertain_visit_threshold=2 \
    --set experiment.experiment_name=alfworld_${MODEL_NAME}_topk3_trichannel_on_111 \
    > ./log/alfworld_${MODEL_NAME}_topk3_trichannel_on_111.log 2>&1 &
  # - Epoch 0: in-dist 43.57% (61/140), out-dist 50.00% (67/134)
  # - Epoch 1: in-dist 44.29% (62/140), out-dist 40.30% (54/134)
  # - Epoch 2: in-dist 45.00% (63/140), out-dist 33.58% (45/134)
  # - Epoch 3: in-dist 49.29% (69/140), out-dist 48.51% (65/134)











################################ thompson sampling ##################### 

nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=5 \
    --set rl_config.tri_channel_enabled=false \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k5_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k5_topk3.log 2>&1 &


nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=10 \
    --set rl_config.tri_channel_enabled=false \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k10_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k10_topk3.log 2>&1 &

nohup python run/run_alfworld.py \
    --config configs/rl_alf_config.yaml \
    --set llm.model=Qwen/Qwen2.5-7B-Instruct \
    --set llm.base_url=http://10.225.68.29:1701/v1 \
    --set experiment.num_sections=20 \
    --set memory.k_retrieve=20 \
    --set rl_config.tri_channel_enabled=false \
    --set rl_config.topk=3 \
    --set rl_config.use_thompson_sampling=true \
    --set experiment.experiment_name=alfworld_ts_${MODEL_NAME}_k20_topk3 \
    > ./log/alfworld_ts_${MODEL_NAME}_k20_topk3.log 2>&1 &