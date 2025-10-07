

2025年8月13日
7:33

 ●  Lift 任务 weight=0.5
 
  预训练指令
 
  1. ShortCut Flow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_mlp_img denoising_steps=20
 
  2. MeanFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_mlp_img
 
  3. ReFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_reflow_mlp_img
 
  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_mlp_img denoising_steps=20
 
  5. ShortCut + InfoNCE Cosine dispersive loss
 
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20
 
  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20
 
  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20
 
  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_dispersive_mlp_img
 
  预训练评估指令
 
  1. ShortCut Flow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=
 
  3. ReFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img_new base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=
 
  微调指令
 
  1. ShortCut Flow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  3. ReFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  微调评估指令
 
  1. ShortCut Flow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=
 
  3. ReFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img_new base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=
 
  ---


2025年8月13日
7:34

 ●  Can 任务 weight=0.5

  预训练指令

  1. ShortCut Flow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_mlp_img

  3. ReFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_mlp_img denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_dispersive_mlp_img

  预训练评估指令

  1. ShortCut Flow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  微调指令

  1. ShortCut Flow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  微调评估指令

  1. ShortCut Flow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  ---


2025年8月13日
7:33

● Square 任务 weight=0.5
 
  预训练指令
 
  1. ShortCut Flow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img denoising_steps=20
 
  2. MeanFlow 基线
 
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_mlp_img
 
  3. ReFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_reflow_mlp_img
 
  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_mlp_img
  denoising_steps=20
 
  5. ShortCut + InfoNCE Cosine dispersive loss
 
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20
 
  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20
 
  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20
 
  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_dispersive_mlp_img
 
  预训练评估指令
 
  1. ShortCut Flow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=
 
  3. ReFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img_new base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=
 
  微调指令
 
  1. ShortCut Flow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  3. ReFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=
 
  微调评估指令
 
  1. ShortCut Flow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  2. MeanFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=
 
  3. ReFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img_new base_policy_path=
 
  4. ShortCut + InfoNCE L2 dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  5. ShortCut + InfoNCE Cosine dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  6. ShortCut + Hinge dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  7. ShortCut + Covariance dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=
 
  8. MeanFlow + dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=



2025年8月13日
7:34

●  Transport 任务 weight=0.5

  预训练指令

  1. ShortCut Flow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow 基线

  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_mlp_img

  3. ReFlow 基线
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_mlp_img
  denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_dispersive_mlp_img

  预训练评估指令

  1. ShortCut Flow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow 基线评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  微调指令

  1. ShortCut Flow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 微调
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  微调评估指令

  1. ShortCut Flow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss 微调评估
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=
