import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import bpy
from mathutils import Vector


def blender_args():
    import sys

    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def apply_transform(obj, spec):
    obj.scale = (spec["scale"], spec["scale"], spec["scale"])
    obj.location = spec["location"]
    obj.rotation_euler = spec["rotation_euler"]


def import_asset(spec):
    path = Path(spec["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing asset: {path}")
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset type: {path}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        obj.name = f"{spec['name']}_{obj.name}"
        apply_transform(obj, spec)
        if obj.type == "MESH" and len(obj.data.polygons) == 0:
            raise ValueError(
                f"{path} contains points but no mesh faces. Export a surface mesh from 2DGS before Blender fusion."
            )
    return imported


def setup_materials():
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.data.materials:
            continue
        mat = bpy.data.materials.new(f"{obj.name}_mat")
        color_attributes = getattr(obj.data, "color_attributes", None)
        if color_attributes and len(color_attributes) > 0:
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            principled = nodes.get("Principled BSDF")
            vertex_color = nodes.new("ShaderNodeVertexColor")
            vertex_color.layer_name = color_attributes[0].name
            links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])
        else:
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(mat)


def create_camera_path(cfg):
    path_cfg = cfg["camera_path"]
    frames = int(path_cfg["frames"])
    radius = float(path_cfg["radius"])
    height = float(path_cfg["height"])
    look_at = Vector(path_cfg["look_at"])
    start_angle = math.radians(float(path_cfg.get("start_angle_deg", 0.0)))
    arc = math.radians(float(path_cfg.get("arc_degrees", 360.0)))

    camera = bpy.data.objects.new("WalkthroughCamera", bpy.data.cameras.new("WalkthroughCamera"))
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    for frame in range(frames):
        if frames > 1:
            angle = start_angle + arc * frame / (frames - 1)
        else:
            angle = start_angle
        camera.location = (
            look_at.x + radius * math.cos(angle),
            look_at.y + radius * math.sin(angle),
            height,
        )
        direction = look_at - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame + 1)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame + 1)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames


def setup_lighting():
    bpy.ops.object.light_add(type="AREA", location=(0, -3, 5))
    light = bpy.context.object
    light.name = "LargeSoftbox"
    light.data.energy = 550
    light.data.size = 5
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.06, 0.065, 0.07, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45


def configure_render(cfg, out):
    render_cfg = cfg["render"]
    scene = bpy.context.scene
    scene.render.resolution_x = int(render_cfg["resolution_x"])
    scene.render.resolution_y = int(render_cfg["resolution_y"])
    scene.render.fps = int(render_cfg["fps"])
    render_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in render_engines else "BLENDER_EEVEE"

    out = out.resolve()
    if out.suffix.lower() in [".mp4", ".mov", ".mkv"]:
        try:
            scene.render.filepath = str(out)
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            return None
        except TypeError as exc:
            print(f"Blender movie output unavailable, falling back to PNG frames: {exc}")

    frame_dir = out.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frame_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    return frame_dir


def encode_frames(frame_dir, out, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"PNG frames saved to {frame_dir}. Install ffmpeg or import the frames into Blender/PR to make a video.")
        return
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out.resolve()),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse 2DGS proxy and mesh assets.")
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parsed = parser.parse_args(blender_args())

    cfg = json.loads(parsed.scene_config.read_text(encoding="utf-8"))
    clear_scene()
    import_asset(cfg["background"])
    for spec in cfg["objects"]:
        import_asset(spec)
    setup_materials()
    setup_lighting()
    create_camera_path(cfg)
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = configure_render(cfg, parsed.out)
    bpy.ops.render.render(animation=True)
    if frame_dir is not None and parsed.out.suffix.lower() in [".mp4", ".mov", ".mkv"]:
        encode_frames(frame_dir, parsed.out, int(cfg["render"]["fps"]))
