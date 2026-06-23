# Experiment Runbook

这份清单按实际执行顺序组织，适合放在 AutoDL、Colab 或实验室服务器上逐步运行。

## 0. Clone and Prepare

```bash
git clone https://github.com/<your-name>/2dgs-aigc-lerobot-final.git
cd 2dgs-aigc-lerobot-final
conda env create -f environment.yml
conda activate cv-final
```

If the server uses NVIDIA driver 470.x / CUDA 11.4, use the compatible environment instead:

```bash
conda env create -f environment.cuda114.yml
conda activate cv-final-cu114
mkdir -p third_party data outputs
```

## 1. External Repositories

```bash
git clone https://github.com/hbb1/2d-gaussian-splatting.git third_party/2d-gaussian-splatting
git clone https://github.com/threestudio-project/threestudio.git third_party/threestudio
git clone https://github.com/guochengqian/Magic123.git third_party/Magic123
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
```

Then install each project according to its official README.

## 2. 3D Vision Pipeline

```bash
python scripts/prepare_video_frames.py --video data/object_a/raw/object_a.mp4 --out data/object_a/images --fps 3 --max-side 1200
python scripts/run_2dgs_pipeline.py --repo third_party/2d-gaussian-splatting --source data/object_a --output outputs/2dgs/object_a --config configs/2dgs/object_a.yaml
python scripts/run_threestudio_asset.py --repo third_party/threestudio --config configs/threestudio/object_b_sds.yaml --output outputs/aigc_assets/object_b
python scripts/run_magic123_asset.py --repo third_party/Magic123 --image data/object_c/object_c_rgba.png --config configs/magic123/object_c.yaml --output outputs/aigc_assets/object_c
python scripts/run_2dgs_pipeline.py --repo third_party/2d-gaussian-splatting --source data/mipnerf360/garden --output outputs/2dgs/background_garden --config configs/2dgs/background_garden.yaml
blender -b -P scripts/fuse_and_render_blender.py -- --scene-config configs/fusion_scene.json --out outputs/fusion/final_walkthrough.mp4
```

## 3. ACT Pipeline

```bash
python scripts/run_lerobot_act.py train --repo third_party/lerobot --config configs/lerobot/act_calvin_b.yaml --output outputs/lerobot/act_env_b
python scripts/run_lerobot_act.py train --repo third_party/lerobot --config configs/lerobot/act_calvin_abc.yaml --output outputs/lerobot/act_env_abc
python scripts/run_lerobot_act.py eval --repo third_party/lerobot --checkpoint outputs/lerobot/act_env_b/checkpoints/best.pt --env D --output outputs/lerobot/eval_env_b_on_d.json
python scripts/run_lerobot_act.py eval --repo third_party/lerobot --checkpoint outputs/lerobot/act_env_abc/checkpoints/best.pt --env D --output outputs/lerobot/eval_env_abc_on_d.json
```

## 4. Figures and Report

```bash
python scripts/plot_training_curves.py --baseline outputs/lerobot/logs/act_env_b.csv --multi-env outputs/lerobot/logs/act_env_abc.csv --out-dir outputs/figures
cd report
latexmk -pdf main.tex
```

## 5. Final Submission

1. Replace every `TODO` in `report/main.tex` with actual measured values.
2. Put representative rendered images in `outputs/figures`.
3. Upload best checkpoints and 3D assets to cloud storage.
4. Update GitHub and weight links in both `README.md` and `report/main.tex`.
5. Submit `report/main.pdf`, public GitHub URL, and model weight URL.
