# MemRL: Self-Evolving Agents via Runtime Reinforcement Learning with Thompson Sampling on Episodic Memory


## Abstract

Autonomous agents must improve from experience while preserving the reasoning ability already encoded in their foundation models. The original **MemRL** framework addresses this stability-plasticity challenge with a non-parametric runtime learning approach: instead of updating model weights, it stores episodic experiences, retrieves relevant memories through a Two-Phase Retrieval mechanism, and uses environmental feedback to reinforce memories that lead to successful behavior. This design turns memory into a plastic component of the agent while keeping the base model stable, enabling continual improvement across tasks without fine-tuning.

This work extends the original MemRL formulation by introducing **Thompson sampling** into memory selection. Rather than relying only on deterministic similarity scores or greedy utility estimates, the agent maintains an uncertainty-aware sampling process over candidate memories. Memories with stronger empirical success are sampled more often, while uncertain or under-explored memories can still be selected when their posterior utility may be high. This exploration-exploitation mechanism reduces premature commitment to noisy early experiences, reallocates retrieval probability toward memories that prove useful through interaction, and improves the effectiveness of episodic memory during runtime learning. Experiments on **ALFWorld** show that adding Thompson sampling strengthens MemRL's original memory-based self-improvement pipeline and yields better agent performance without any model weight updates.

## Thompson Sampling Extension

The original MemRL pipeline already improves over passive retrieval by using environmental feedback to identify useful memories. However, memory selection can still become overly greedy: early successful memories may dominate the context even when they are only locally useful, while newer or less-tested memories may be ignored before the agent has enough evidence about them. Thompson sampling addresses this by treating memory selection as an uncertainty-aware decision problem, balancing reuse of proven memories with exploration of candidates that may become valuable after more interaction.

- `rl_config.use_thompson_sampling` enables the incremental memory-selection policy contributed in this repo. When enabled, retrieved memories are selected with Thompson sampling instead of relying only on deterministic ranking.
- `rl_config.topk` controls how many candidate memories are sampled from the retrieval pool and passed into the agent context. Larger values expose the agent to more past experiences, while smaller values make memory use more selective.
- The sampling unit is an episodic memory candidate produced by MemRL retrieval. Each candidate's observed success/failure feedback updates its estimated utility, so useful memories become more likely to be reused while uncertain memories can still be explored.

## Framework Overview

Click to open the PDF:

[![MemRL framework overview](framework_overview.png)](framework_overview.pdf)

Files:
- `framework_overview.png` (preview)
- `framework_overview.pdf` (vector)

## Installation


```bash
conda create -n memoryrl python=3.10 -y

conda activate memoryrl

pip install -U pip

pip install -r requirements.txt
```


## Running the ALFWorld Benchmark

The run writes logs under `logs/` and results under `results/` (configurable via `experiment.output_dir`).

Run:

```bash
python run/run_alfworld.py \
--config configs/rl_alf_config.yaml \
--set llm.model=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--set llm.base_url=http://10.225.68.16:1700/v1 \
--set experiment.num_sections=10 \
--set memory.k_retrieve=10 \
--set rl_config.topk=4 \
--set rl_config.use_thompson_sampling=true \
--set experiment.experiment_name=alfworld_with_ts_Qwen3-30B-A3B-Instruct-2507-FP8_k10_topk4
```

Important notes:

- You must install ALFWorld and prepare its data according to the [ALFWorld](https://github.com/alfworld/alfworld) setup.
- This repo expects an ALFWorld environment config at:
  `configs/envs/alfworld.yaml`
  (Provided).
- Few-shot examples are expected at `data/alfworld/alfworld_examples.json` (Provided, same as [ReAct](https://github.com/ysymyth/ReAct)) (configurable via `experiment.few_shot_path`).



## Project Layout

- `memrl/`: main library code (MemoryService, runners, providers, tracing)
- `run/`: benchmark entrypoints (`run_bcb.py`, `run_llb.py`, `run_alfworld.py`, `run_hle.py`)
- `configs/`: benchmark configs
- `3rdparty/`: vendored benchmark repos (BigCodeBench, LifelongAgentBench)

## Citation
This repo is forked from original project:
```bibtex
@misc{zhang2026memrlselfevolvingagentsruntime,
  title         = {MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory},
  author        = {Shengtao Zhang and Jiaqian Wang and Ruiwen Zhou and Junwei Liao and Yuchen Feng and Weinan Zhang and Ying Wen and Zhiyu Li and Feiyu Xiong and Yutao Qi and Bo Tang and Muning Wen},
  year          = {2026},
  eprint        = {2601.03192},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.03192},
}
```
