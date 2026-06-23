# Finalization Runbook

This checklist starts after Object C finishes training.

## 1. Archive Object C

```bash
find ~/cv-final/outputs/aigc_assets/object_c_3k -name "*.ckpt"
find ~/cv-final/outputs/aigc_assets/object_c_3k -type f \( -name "*.png" -o -name "*.mp4" \) | tail -30
```

Export the coarse Magic123 checkpoint with `system.exporter_type=mesh-exporter`, then keep the OBJ, MTL, texture image, checkpoint, and test video together.

## 2. Locate 2DGS Surface Meshes

The Blender fusion route requires triangle meshes, not raw Gaussian point-cloud PLY files.

```bash
find ~/cv-final/outputs/2dgs/object_a_30k -type f \( -iname "*mesh*.ply" -o -iname "*fuse*.ply" \)
find ~/cv-final/outputs/2dgs/background_counter_30k -type f \( -iname "*mesh*.ply" -o -iname "*fuse*.ply" \)
python ~/cv-final/third_party/2d-gaussian-splatting/render.py -h
```

If no mesh exists, use the mesh/TSDF extraction options shown by `render.py -h`. Preserve the original Gaussian PLY files as model weights, but use extracted surface meshes for Blender.

## 3. Assemble Canonical Fusion Assets

Copy downloaded assets into this layout on the local machine:

```text
outputs/fusion/assets/
├── background_counter_mesh.ply
├── object_a_mesh.ply
├── object_b/
│   ├── model.obj
│   ├── model.mtl
│   └── texture_kd.png
└── object_c/
    ├── model.obj
    ├── model.mtl
    └── texture_kd.png
```

Keep each OBJ, MTL, and texture in the same directory.

## 4. Blender Fusion

Edit object scales and locations in `configs/fusion_scene.json`, then render:

```bash
blender -b -P scripts/fuse_and_render_blender.py -- \
  --scene-config configs/fusion_scene.json \
  --out outputs/fusion/final_walkthrough.mp4
```

First render a short 24-frame preview by temporarily setting `camera_path.frames` to 24. After placement is correct, restore 180 frames for the final video.

## 5. Evidence to Preserve

- Object A: source photo montage, COLMAP sparse reconstruction, 2DGS render, checkpoint/PLY, runtime.
- Background: 3k and 30k renders, `results.json`, PSNR/SSIM/LPIPS, runtime.
- Object B: prompt, 1k/3k render, checkpoint, exported textured mesh, runtime.
- Object C: source and RGBA image, Magic123 render, checkpoint, exported textured mesh, runtime.
- Fusion: Blender screenshot, transform configuration, final MP4.
- ACT: two real training logs, D evaluation JSON, checkpoints, curve exports.

## 6. Submission

1. Replace every `TODO` in `report/main.tex` with measured values or explicitly mark an experiment as incomplete.
2. Compile with `latexmk -xelatex main.tex`.
3. Push the public GitHub repository without datasets, framework copies, COLMAP binaries, or model weights.
4. Upload checkpoints, Gaussian PLY files, textured meshes, and video to cloud storage.
5. Put both public links on the first page of the report and in `README.md`.
