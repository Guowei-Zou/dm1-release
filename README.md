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

- **Single-Step Inference** – Flow-based controller executes in `<DM1_INFERENCE_LATENCY>` per timestep.
- **Dispersive Loss Family** – Plug-and-play InfoNCE / Cosine / Hinge regularizers prevent feature collapse.
- **Vision-Ready** – Transformer encoder handles multi-view RGB observations out-of-the-box.
- **Benchmarks & Real Robots** – Validated on `<PRIMARY_BENCHMARK>` and `<REAL_ROBOT_PLATFORM>`.
- **Modular Configs** – YAML-driven experiment recipes across pre-training, evaluation, and ablations.

---

## 🚀 Quick Start

### 1. Clone & Environment Setup

```bash
git clone <DM1_RELEASE_REPO_URL>
cd dm1-release
conda create -n dm1 python=<PYTHON_VERSION> -y
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

### 2. External Dependencies

| Environment Suite | Requirement | Notes |
| ----------------- | ----------- | ----- |
| Robomimic | MuJoCo `<MUJOCO_VERSION>` | see `installation/install_mujoco.md`
| D3IL | `<D3IL_DEPENDENCY>` | see `installation/install_d3il.md`
| Furniture | Isaac Gym `<ISAAC_VERSION>` | see `installation/install_furniture.md`

Set shared paths and logging endpoints:
```bash
source script/set_path.sh  # defines <DATA_ROOT>, <LOG_ROOT>, <WANDB_ENTITY>
```

---

## 📦 Datasets & Checkpoints

- Demonstration bundles: `<DM1_DATASET_URL>`  
  Downloaded automatically to `<DATA_ROOT>` when launching pre-training.
- Pretrained checkpoints: [Google Drive](https://drive.google.com/drive/folders/1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8?usp=sharing)  
  Mirrors the latest DM1 pretraining weights; sync into `<LOG_ROOT>` for evaluation scripts.
- Evaluation statistics: [Google Drive](https://drive.google.com/drive/folders/1OjsDdBxOVy2x77cwsL1V2T7ey_slp7nj?usp=sharing)  
  Contains aggregated `.npz` metrics corresponding to the checkpoints above.

To reuse custom data, drop trajectories under `<CUSTOM_DATA_DIR>` and update `cfg/<ENV_GROUP>/pretrain/<TASK>.yaml`.

---

## 🧪 Running DM1

### Dispersive Pre-Training (Image-Based)
```bash
python script/run.py \
  --config-dir=cfg/robomimic/pretrain/<TASK_NAME> \
  --config-name=pre_dm1_mlp_img_dispersive \
  denoising_steps=<DM1_DENOISE_STEPS> \
  dispersive.loss_type=<DISPERSIVE_VARIANT> \
  dispersive.weight=<DISPERSIVE_WEIGHT>
```
Available `<TASK_NAME>`: `lift`, `can`, `square`, `transport`.

### State-Based Variants
```bash
python script/run.py \
  --config-dir=cfg/<ENV_GROUP>/pretrain/<TASK_NAME> \
  --config-name=pre_dm1_mlp_state_dispersive
```
`<ENV_GROUP>` can be `gym`, `d3il`, or `kitchen` (Franka Kitchen).

### Evaluation & Rollouts
```bash
python script/run.py \
  --config-dir=cfg/robomimic/eval/<TASK_NAME> \
  --config-name=eval_dm1_mlp_img \
  base_policy_path=<CHECKPOINT_PATH>
```
Metrics and plots are stored in `dm1_pretraining_eval_results/`.

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

| Setting | Metric | DM1 (w/ dispersive) | Baseline |
| ------- | ------ | ------------------- | -------- |
| Robomimic Lift | success @ 2000 iters | `<DM1_LIFT_SUCCESS>` | `<BASELINE_LIFT_SUCCESS>` |
| Robomimic Can | success @ 2000 iters | `<DM1_CAN_SUCCESS>` | `<BASELINE_CAN_SUCCESS>` |
| Inference Latency | per action | `<DM1_LATENCY>` | `<BASELINE_LATENCY>` |

Replace placeholders with your measurements after running the provided scripts in `tools/eval/`.

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
├── docs/                          # extended documentation
└── installation/                  # environment setup guides
```

---

## 🔬 Research Notes

- **Problem**: Diffusion-style policies trained from demonstrations often collapse to narrow manifolds when regularization is weak.
- **Solution**: DM1 introduces dispersive loss variants that maintain feature diversity while retaining meanflow efficiency.
- **Impact**: Demonstrated gains on `<PRIMARY_BENCHMARK>` plus transfer to `<REAL_ROBOT_PLATFORM>` with minimal tuning.

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
- [Diffusion Policy](<DIFFUSION_POLICY_URL>)
- [MeanFlow / Shortcut Models](<MEANFLOW_REPO_URL>)
- [Robomimic Benchmark](<ROBOMIMIC_URL>)
- [DPPO](<DPPO_REPO_URL>)

See `THIRD_PARTY_LICENSES.md` for complete dependency attributions.

---

## 📄 License

Released under the `<DM1_LICENSE>` License. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

- Submit issues: `<DM1_ISSUES_URL>`  
- Join discussions: `<DM1_COMMUNITY_LINK>`  
- Email: `<DM1_CONTACT_EMAIL>`

---

*This release focuses on pre-training and evaluation workflows. PPO-style online fine-tuning is tracked in `<DM1_FUTURE_WORK_URL>`.*
