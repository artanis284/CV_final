# 2DGS-AIGC Fusion and LeRobot ACT Generalization

本仓库用于完成计算机视觉综合项目的两个题目：

- 题目一：基于 2D Gaussian Splatting 与 AIGC 的多源 3D 资产生成、真实场景融合与漫游渲染。
- 题目二：基于 LeRobot 的 ACT 策略跨环境泛化实验。

仓库采用“外部框架 + 本项目配置与脚本”的组织方式。2DGS、threestudio、Magic123 与 LeRobot 均保持官方仓库安装，本仓库负责数据目录规范、实验配置、运行入口、融合脚本、日志可视化和报告材料。

## Repository

```text
GitHub: https://github.com/artanis284/CV_final
Weights: https://drive.google.com/drive/folders/1Mn-dgOzEQUyMsfwXLPx0SZvPlfBLO5ii
```

## Directory Layout

```text
.
├── configs/
│   ├── 2dgs/
│   ├── threestudio/
│   ├── magic123/
│   └── lerobot/
├── data/
│   ├── object_a/
│   ├── object_c/
│   ├── mipnerf360/
│   └── calvin/
├── outputs/
│   ├── 2dgs/
│   ├── aigc_assets/
│   ├── fusion/
│   ├── lerobot/
│   └── figures/
├── report/
├── scripts/
└── environment.yml
```

## Requirements

建议在 AutoDL、Colab Pro、Kaggle 或本地 Linux GPU 环境运行。推荐显存不低于 16 GB；显存不足时，将图像最长边降低到 800 或 1024。

```bash
conda env create -f environment.yml
conda activate cv-final
```

For older CUDA 11.4 drivers, such as NVIDIA driver 470.x, use:

```bash
conda env create -f environment.cuda114.yml
conda activate cv-final-cu114
```

需要额外克隆的外部仓库：

```bash
mkdir -p third_party
git clone https://github.com/hbb1/2d-gaussian-splatting.git third_party/2d-gaussian-splatting
git clone https://github.com/threestudio-project/threestudio.git third_party/threestudio
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
```

安装外部框架依赖时请优先遵循各官方 README。本仓库脚本通过命令行参数指向这些仓库位置。

## Data Preparation

### Object A: Real Multi-view Reconstruction

1. 使用手机围绕真实物体拍摄一段 30 到 90 秒视频，保持曝光稳定，背景纹理丰富。
2. 抽帧到 `data/object_a/images`：

```bash
python scripts/prepare_video_frames.py --video data/object_a/raw/object_a.mp4 --out data/object_a/images --fps 3 --max-side 1200
```

3. 运行 COLMAP 并转换为 2DGS 可用格式：

```bash
python scripts/run_2dgs_pipeline.py \
  --repo third_party/2d-gaussian-splatting \
  --source data/object_a \
  --output outputs/2dgs/object_a \
  --config configs/2dgs/object_a.yaml
```

### Object B: Text-to-3D

使用 threestudio + SDS Loss 生成文本资产：

```bash
python scripts/run_threestudio_asset.py \
  --repo third_party/threestudio \
  --config configs/threestudio/object_b_sds.yaml \
  --output outputs/aigc_assets/object_b
```

默认 Prompt 为：

```text
a small ceramic robot planter, glossy white body, tiny green succulent, studio lighting, high quality
```

### Object C: Single-image-to-3D

1. 拍摄单张真实物体照片，去背景后保存为 `data/object_c/object_c_rgba.png`。
2. 使用 threestudio 内置 Magic123 coarse 配置生成 3D 模型：

```bash
python third_party/threestudio/launch.py \
  --config third_party/threestudio/configs/magic123-coarse-sd.yaml \
  --train --gpu 0 \
  data.image_path=data/object_c/object_c_rgba.png \
  system.prompt_processor.prompt="a dark gray stainless steel water bottle with a black loop cap"
```

### Background Scene: Mip-NeRF 360 Counter

选择 garden、bicycle 或 counter 场景，放到 `data/mipnerf360/<scene>`。示例：

```bash
python scripts/run_2dgs_pipeline.py \
  --repo third_party/2d-gaussian-splatting \
  --source data/mipnerf360/counter \
  --output outputs/2dgs/background_counter_30k \
  --config configs/2dgs/background_counter.yaml
```

## Scene Fusion and Rendering

本项目采用统一为 Mesh 后在 Blender 中融合渲染的路线：

1. 2DGS 结果导出为点云或 mesh-like surfel proxy。
2. threestudio 与 Magic123 资产导出为带纹理 mesh。
3. 通过 `fusion_scene.json` 统一配置尺度、旋转和平移。
4. Blender 加载背景 proxy 与三个物体 mesh，生成多视角漫游视频。

```bash
blender -b -P scripts/fuse_and_render_blender.py -- \
  --scene-config configs/fusion_scene.json \
  --out outputs/fusion/final_walkthrough.mp4
```

如需代码级拼接高斯表示，可使用 `scripts/sample_mesh_to_gaussians.py` 将 AIGC mesh 均匀采样为带颜色点云，再转换为 2DGS 初始化点云。

## LeRobot ACT 实验

ACT 实验在本地 Windows 笔记本 RTX 4060 Laptop GPU 上完成。由于官方 CALVIN 划分体量较大，联合模型采用轻量采样子集：从环境 A、B、C 中各取约 1k 条 episode。所有数据集均转换为 LeRobot v3.0 格式，并统一重命名为 ACT 兼容字段：

```text
image       -> observation.images.image
wrist_image -> observation.images.wrist_image
state       -> observation.state
actions     -> action
```

### 训练环境 B 基线模型

```powershell
lerobot-train `
  --policy.type=act `
  --policy.push_to_hub=false `
  --dataset.repo_id=local/calvin_splitB_act `
  --dataset.root=D:\cv_final_task2\datasets\calvin-lerobot\splitB_act `
  --dataset.use_imagenet_stats=false `
  --batch_size=2 `
  --steps=1000 `
  --output_dir=D:\cv_final_task2\outputs\act_env_b_local_1k `
  --wandb.enable=false
```

### 训练 A+B+C 多环境联合模型

```powershell
lerobot-train `
  --policy.type=act `
  --policy.push_to_hub=false `
  --dataset.repo_id=local/calvin_splitABC_act_1k_each `
  --dataset.root=D:\cv_final_task2\datasets\calvin-lerobot\splitABC_act_1k_each `
  --dataset.use_imagenet_stats=false `
  --batch_size=2 `
  --steps=1000 `
  --output_dir=D:\cv_final_task2\outputs\act_env_abc_local_1k `
  --wandb.enable=false
```

### 在环境 D 上进行 Zero-shot 测试

```powershell
python D:\cv_final_task2\lerobot\eval_act_l1.py `
  --dataset-root D:\cv_final_task2\datasets\calvin-lerobot\splitD_act `
  --checkpoint D:\cv_final_task2\outputs\act_env_b_local_1k\checkpoints\001000\pretrained_model `
  --repo-id local/calvin_splitD_act `
  --batch-size 4 `
  --max-batches 200

python D:\cv_final_task2\lerobot\eval_act_l1.py `
  --dataset-root D:\cv_final_task2\datasets\calvin-lerobot\splitD_act `
  --checkpoint D:\cv_final_task2\outputs\act_env_abc_local_1k\checkpoints\001000\pretrained_model `
  --repo-id local/calvin_splitD_act `
  --batch-size 4 `
  --max-batches 200
```

环境 D 上测得的 zero-shot Action L1 结果如下：

| 模型 | 训练数据 | 训练步数 | Action L1 |
|---|---:|---:|---:|
| ACT-B | 仅环境 B | 1,000 | 0.360068 |
| ACT-ABC | 环境 A/B/C 采样子集 | 1,000 | 0.310211 |

联合模型将环境 D 上的 zero-shot Action L1 误差降低约 13.85%。从每 50 step 记录一次的控制台日志中解析得到，1k step 时 ACT-B 的训练 loss 为 2.917，ACT-ABC 的训练 loss 为 2.934。

