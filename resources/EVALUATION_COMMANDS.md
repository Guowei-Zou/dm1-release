# DM1 Evaluation Commands - Quick Reference

**直接复制下面的命令即可运行评估**

---

## 📋 Lift Task

### w_0p1 (Noise Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/lift/lift_w0p1_08_meanflow_dispersive.pt
```

### w_0p5 (Noise Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/lift/lift_w0p5_08_meanflow_dispersive.pt
```

### w_0p9 (Noise Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/lift/lift_w0p9_08_meanflow_dispersive.pt
```


---

## 📋 Can Task

### w_0p1 (Noise Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/can/can_w0p1_08_meanflow_dispersive.pt
```

### w_0p5 (Noise Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/can/can_w0p5_08_meanflow_dispersive.pt
```

### w_0p9 (Noise Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/can/can_w0p9_08_meanflow_dispersive.pt
```

---

## 📋 Square Task

### w_0p1 (Noise Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/square/square_w0p1_08_meanflow_dispersive.pt
```

### w_0p5 (Noise Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/square/square_w0p5_08_meanflow_dispersive.pt
```

### w_0p9 (Noise Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/square/square_w0p9_08_meanflow_dispersive.pt
```

---

## 📋 Transport Task

### w_0p1 (Noise Weight = 0.1)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p1/transport/transport_w0p1_08_meanflow_dispersive.pt
```

### w_0p5 (Noise Weight = 0.5)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p5/transport/transport_w0p5_08_meanflow_dispersive.pt
```

### w_0p9 (Noise Weight = 0.9)

```bash
# 01 - ShortCut Flow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_01_shortcut_flow_baseline.pt

# 02 - MeanFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_02_meanflow_baseline.pt

# 03 - ReFlow Baseline
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_03_reflow_baseline.pt

# 04 - ShortCut + InfoNCE L2
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_04_shortcut_infonce_l2.pt

# 05 - ShortCut + InfoNCE Cosine
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_05_shortcut_infonce_cosine.pt

# 06 - ShortCut + Hinge
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_06_shortcut_hinge.pt

# 07 - ShortCut + Covariance
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_07_shortcut_covariance.pt

# 08 - MeanFlow + Dispersive
python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=dm1_pretraining_checkpoints/w_0p9/transport/transport_w0p9_08_meanflow_dispersive.pt
```

---

## 📝 Quick Reference

**Results Location:** `dm1_pretraining_eval_results/`

**Experiment Index:**
- 01: ShortCut Flow Baseline
- 02: MeanFlow Baseline
- 03: ReFlow Baseline
- 04: ShortCut + InfoNCE L2
- 05: ShortCut + InfoNCE Cosine
- 06: ShortCut + Hinge
- 07: ShortCut + Covariance
- 08: MeanFlow + Dispersive
