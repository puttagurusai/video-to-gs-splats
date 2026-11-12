"""Blender: render the scene in an open .blend from the held-out camera poses in views.json (flat texture colours).

  blender --background scene.blend --python scripts/render_views.py -- --views export/room_relief/views.json --out-prefix export/room_eval_relief

Writes <prefix>_<frame>.png (RGBA; transparent where there is no geometry). Compare with scripts/eval_views.py.
"""
import argparse
import json
import os
import sys

import bpy
from mathutils import Matrix


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="render size relative to the original frame")
    a = ap.parse_args(argv)
    a.views, a.out_prefix = os.path.abspath(a.views), os.path.abspath(a.out_prefix)
    views = json.load(open(a.views))

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light, sh.color_type = "FLAT", "TEXTURE"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"  # no tone mapping, so colours are comparable to the video frames
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    cam_data = bpy.data.cameras.new("c")
    cam_data.clip_start, cam_data.clip_end, cam_data.sensor_fit = 0.01, 200, "AUTO"
    cam = bpy.data.objects.new("c", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for v in views:
        for o in scene.objects:  # views with a "piece" render only that object (camera-mapping sanity check)
            if o.type == "MESH":
                o.hide_render = bool(v.get("piece")) and not o.name.startswith(v["piece"])
        R, loc = v["rotation"], v["location"]
        cam.matrix_world = Matrix(((R[0][0], R[0][1], R[0][2], loc[0]), (R[1][0], R[1][1], R[1][2], loc[1]),
                                   (R[2][0], R[2][1], R[2][2], loc[2]), (0, 0, 0, 1)))
        cam_data.angle = v["angle"]
        scene.render.resolution_x = int(round(v["W"] * a.scale))
        scene.render.resolution_y = int(round(v["H"] * a.scale))
        scene.render.resolution_percentage = 100
        scene.render.filepath = f"{a.out_prefix}_{os.path.splitext(v['name'])[0]}.png"
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)


if __name__ == "__main__":
    main()
