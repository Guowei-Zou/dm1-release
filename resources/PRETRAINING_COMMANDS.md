# DM1 Pre-training Commands - Quick Reference

**直接复制下面的命令即可启动预训练（需先激活 `dm1` 环境并完成 `pip install -e .[robomimic]`）**

---

> 说明：
> - 01–03 为无分散正则的基线（ShortCut / MeanFlow / ReFlow）
> - 04–08 使用分散正则，需要按权重 `w` 调整 `model.dispersive_loss_weight` 与 `dispersive_weight_name`
> - 每条命令默认使用配置文件中的其他超参，可根据实际需求额外覆盖（如 `seed`、`device` 等）

## 📋 Lift Task

### w_0p1 (Dispersive Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1
```

### w_0p5 (Dispersive Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5
```

### w_0p9 (Dispersive Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9
```

---

## 🥫 Can Task

### w_0p1 (Dispersive Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1
```

### w_0p5 (Dispersive Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5
```

### w_0p9 (Dispersive Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9
```

---

## 🔳 Square Task

### w_0p1 (Dispersive Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1
```

### w_0p5 (Dispersive Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5
```

### w_0p9 (Dispersive Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9
```

---

## 🚚 Transport Task

### w_0p1 (Dispersive Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p1 model.dispersive_loss_weight=0.1
```

### w_0p5 (Dispersive Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p5 model.dispersive_loss_weight=0.5
```

### w_0p9 (Dispersive Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_mlp_img

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_mlp_img

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp_img

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_cosine_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_hinge_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_covariance_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_dispersive_mlp_img dispersive_weight_name=0p9 model.dispersive_loss_weight=0.9
```

---

## 📝 Quick Reference

- 日志与模型默认写入 `${REINFLOW_LOG_DIR}/robomimic/pretrain/...`
- 如果希望与 release 中的 `w_0pX` 目录保持一致，可在命令中额外覆盖 `hydra.run.dir`
- 训练完成后可将 `checkpoint/state_XXXX.pt` 移动或复制到 `dm1_pretraining_checkpoints/w_0pX/<task>/` 以便评估脚本直接引用

