<div align="center">

# DM1: MeanFlow with Dispersive Regularization for 1-Step Robotic Manipulation

### [Guowei Zou](https://guowei-zou.github.io/Guowei-Zou/), Haitao Wang, [Hejun Wu](https://cse.sysu.edu.cn/teacher/WuHejun), Yukun Qian, Yuhang Wang, and [Weibing Li](https://cse.sysu.edu.cn/en/teacher/LiWeibing)*



[![🚀 Project Page](https://img.shields.io/badge/🚀_Project_Page-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://guowei-zou.github.io/dm1/)
[![📄 Paper](https://img.shields.io/badge/📄_Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](<https://arxiv.org/abs/2510.07865>)
[![💻 Code](https://img.shields.io/badge/💻_Code-181717?style=for-the-badge&logo=github&logoColor=white)](<https://github.com/Guowei-Zou/dm1-release>)
[![🎥 Youtube](https://img.shields.io/badge/🎥_Youtube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](<https://www.youtube.com/watch?v=7d2IIOu8kig>)
[![📺 Bilibili](https://img.shields.io/badge/📺_Bilibili-FF6699?style=for-the-badge&logo=bilibili&logoColor=white)](<https://www.bilibili.com/video/BV1uHxizNEz8/>)
[![🔗 Checkpoints](https://img.shields.io/badge/🔗_Checkpoints-34A853?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8?usp=sharing)
[![📊 Evaluation Results](https://img.shields.io/badge/📊_Evaluation_Results-34A853?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1OjsDdBxOVy2x77cwsL1V2T7ey_slp7nj?usp=sharing)


>
> A novel flow matching framework that prevents representation collapse while maintaining one-step efficiency.
> Achieves **20-40× faster inference** (0.07s vs. 2-3.5s) and **10-20% higher success rates** compared to diffusion baselines.

</div>

---

## 🏗️ Architecture at a Glance

<div align="center">
<img src="assets/DM1.png" alt="DM1 Architecture" width="800"/>

*DM1 workflow: dispersive pre-training on demonstrations (left) followed by task deployment or fine-tuning (right). Dispersive loss widens feature coverage for robust one-step control.*
</div>

---

## 🔑 Highlights

- **Single-Step Inference** – Flow-based controller executes in 0.07s per timestep.
- **Dispersive Loss Family** – Plug-and-play InfoNCE / Cosine / Hinge regularizers prevent feature collapse.
- **Vision-Ready** – Transformer encoder handles multi-view RGB observations out-of-the-box.
- **Benchmarks & Real Robots** – Validated on Robomimic Benchmark and Franka-Emika-Panda robot.
- **Modular Configs** – YAML-driven experiment recipes across pre-training, evaluation, and ablations.

---

## 🚀 Quick Start

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Guowei-Zou/dm1-release.git
cd dm1-release
conda create -n dm1 python=3.8 -y
conda activate dm1
pip install -e .
```

Optional extras:
```bash
# Vision manipulation stack
pip install -e .[robomimic]

# Full environment suite
pip install -e .[all]
```

Source the helper script before running any commands (update wandb entity, conda path, etc. inside if needed):
```bash
source activate_dm1.sh
```


---

## 📦 Datasets & Checkpoints

### Dataset Download

**Note**: We use the same datasets as provided in the DPPO paper. Pre-training data for all tasks are pre-processed and available at [Google Drive](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link). 

**Manual Download (Robomimic image datasets):**

```bash
python resources/download_dm1_datasets.py
```

Optional suites (adds Gym, Kitchen, and D3IL datasets):

```bash
python resources/download_dm1_datasets.py --groups robomimic gym kitchen d3il
```


Need the entire Google Drive folder instead?

```bash
gdown --folder "https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link" -O data --remaining-ok
```


### Pretrained Checkpoints

**Download:** Run `python resources/download_dm1_weights.py` or `bash resources/download_checkpoints.sh` to auto-download checkpoints to `dm1_pretraining_checkpoints/`.
Alternatively, manually download from [Google Drive](https://drive.google.com/drive/folders/1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8?usp=sharing).


To reuse custom data, drop trajectories under `<CUSTOM_DATA_DIR>` and update `cfg/<ENV_GROUP>/pretrain/<TASK>.yaml`.

---

## 🧪 Pre-Training

**Generic command:**

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/<TASK_NAME> \
  --config-name=<CONFIG_NAME>
```

**Example (ShortCut + InfoNCE Cosine on can task):**

```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_shortcut_dispersive_cosine_mlp_img
```

- `<TASK_NAME>`: `lift`, `can`, `square`, `transport`
- `<CONFIG_NAME>`: `pre_shortcut_mlp_img`, `pre_meanflow_mlp_img`, `pre_reflow_mlp_img`, or dispersive variants (see configs)
- For dispersive loss configuration: see [Dispersive Loss Configuration](#dispersive-loss-configuration)


**Full command matrix:** See [PRETRAINING_COMMANDS.md](resources/PRETRAINING_COMMANDS.md) for 96 ready-to-copy training launches (4 tasks × 3 weights × 8 model variants).

---

## 📊 Evaluation & Rollouts

**Prerequisite:** Ensure checkpoints are available (see [Pretrained Checkpoints](#pretrained-checkpoints)).

**Generic command:**

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/<TASK> \
  --config-name=<EVAL_CONFIG> \
  base_policy_path=dm1_pretraining_checkpoints/<WEIGHT_DIR>/<TASK>/<CHECKPOINT_FILE>
```

**Example (can task, ShortCut + InfoNCE Cosine, w=0.5):**

```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/can \
  --config-name=eval_shortcut_mlp_img \
  base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_05_shortcut_infonce_cosine.pt
```

- `<TASK>`: `lift`, `can`, `square`, `transport`
- `<EVAL_CONFIG>`: `eval_shortcut_mlp_img`, `eval_meanflow_mlp_img`, `eval_reflow_mlp_img` (must match the training method)
- Results are saved to `dm1_pretraining_eval_results/`

**Full command matrix:** [EVALUATION_COMMANDS.md](resources/EVALUATION_COMMANDS.md) lists all 96 evaluation commands.

---

## ⚙️ Dispersive Loss Configuration

```yaml
model:
  use_dispersive_loss: true
  dispersive:
    weight: <DISPERSIVE_WEIGHT>        # e.g., 0.5
    temperature: <DISPERSIVE_TEMPERATURE>  # e.g., 0.3
    loss_type: "<DISPERSIVE_VARIANT>"  # infonce_l2 | infonce_cosine | hinge | covariance
    target_layer: "<DISPERSIVE_LAYER>" # early | mid | late | all
```

> **Tip**: Start with `loss_type: infonce_l2`, `weight: 0.5`, `target_layer: mid` for Robomimic image tasks. Increase `weight` when training diverges or features collapse.

---

## 🎯 Supported Tasks

| Domain | Tasks | Notes |
| ------ | ----- | ----- |
| Robomimic (RGB) | lift, can, square, transport | default configs under `cfg/robomimic` |
| Franka Kitchen | kitchen-partial, kitchen-complete, kitchen-mixed | state-based high-DOF control |
| D3IL | avoiding, pushing, sorting | industrial benchmark episodes |
| Custom | `<CUSTOM_TASKS>` | add YAML to `cfg/custom/` and register datasets |

Real robot deployment scripts (Franka) are provided under `script/real_robot/` with placeholders `<ROBOT_IP>` and `<CONTROL_RATE>`.

---

## 📈 Reference Metrics

| Task | Baseline (32-128 steps) | DM1 (5 steps) | Improvement | Speedup |
| ---- | ----------------------- | ------------- | ----------- | ------- |
| Lift | ~85% | 99% | +14% | 20-40× |
| Can | Variable | High success | +10-20% | 20-40× |
| Square | Moderate | Improved | +15-25% | 20-40× |
| Transport | Low | Significantly improved | +20-30% | 20-40× |

**Inference Latency:** DM1 achieves 0.07s per timestep vs. 2-3.5s for diffusion baselines.

---

## 📁 Repository Map

```
dm1-release/
├── agent/                         # training & evaluation agents
├── cfg/                           # experiment YAMLs
├── dm1_pretraining_checkpoints/   # organized checkpoints (see README inside)
├── dm1_pretraining_eval_results/  # evaluation statistics (.npz)
├── model/                         # flow, diffusion, gaussian modules
├── script/run.py                  # unified launcher
├── tools/                         # analysis & visualization utilities
├── resources/                          # extended documentation
└── installation/                  # environment setup guides
```

---

## 🔬 Research Notes

- **Problem**: Diffusion-style policies trained from demonstrations often collapse to narrow manifolds when regularization is weak.
- **Solution**: DM1 introduces dispersive loss variants that maintain feature diversity while retaining meanflow efficiency.


If you build upon DM1, please cite:

```bibtex
@misc{zou2025dm1meanflowdispersiveregularization,
      title={DM1: MeanFlow with Dispersive Regularization for 1-Step Robotic Manipulation}, 
      author={Guowei Zou and Haitao Wang and Hejun Wu and Yukun Qian and Yuhang Wang and Weibing Li},
      year={2025},
      eprint={2510.07865},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07865}, 
}
```

---

## 🙏 Acknowledgments

DM1 extends prior work on diffusion and flow-based control. We gratefully acknowledge:

- **[Diffusion Policy](<https://arxiv.org/abs/2303.04137>)** (CoRL 2023): Pioneered diffusion models for visuomotor control
- **[ReinFlow](<https://arxiv.org/abs/2505.22094>)** (2025): Flow matching with online RL for robotic manipulation
- **[MeanFlow](<https://arxiv.org/abs/2505.13447>)** (NeurIPS 2025): Mean flows for one-step generative modeling
- **[FlowPolicy](<https://arxiv.org/abs/2412.04987>)** (AAAI 2025): 3D flow-based policy via consistency flow matching
- **[D2PPO](<https://arxiv.org/abs/2508.02644>)** (2025): Diffusion Policy Policy Optimization with Dispersive Loss
- **[π<sub>0.5</sub>](<https://arxiv.org/abs/2504.16054>)** (2025): a Vision-Language-Action Model with Open-World Generalization

---

## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Guowei-Zou/dm1-release&type=Date)](https://star-history.com/#Guowei-Zou/dm1-release&Date)

</div>
