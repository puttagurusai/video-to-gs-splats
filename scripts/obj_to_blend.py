"""Blender helper: import an OBJ (Z up, Y forward; one or many textured objects), pack textures, save .blend, export GLB.

  blender --background --python scripts/obj_to_blend.py -- --input export/room_tex.obj --blend export/room_tex.blend --glb export/room_tex.glb
"""
import argparse
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--glb", default="")
    ap.add_argument("--name", default="Scan")
    a = ap.parse_args(argv)
    # Blender resolves relative paths against its own folder, not yours: always use absolute paths
    a.input, a.blend = os.path.abspath(a.input), os.path.abspath(a.blend)
    a.glb = os.path.abspath(a.glb) if a.glb else ""

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=a.input, forward_axis="Y", up_axis="Z")
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if len(objs) == 1:
        objs[0].name = a.name
    print("objects:", len(objs), "faces:", sum(len(o.data.polygons) for o in objs),
          "uv layers ok:", all(len(o.data.uv_layers) >= 1 for o in objs),
          "materials:", len({m.name for o in objs for m in o.data.materials if m}))
    for o in objs:  # photo-baked look: high roughness, no specular
        for m in o.data.materials:
            if m and m.use_nodes:
                b = m.node_tree.nodes.get("Principled BSDF")
                if b:
                    b.inputs["Roughness"].default_value = 1.0
                    if "Specular IOR Level" in b.inputs:
                        b.inputs["Specular IOR Level"].default_value = 0.0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=a.blend)
    print("saved", a.blend)
    if a.glb:
        bpy.ops.export_scene.gltf(filepath=a.glb, export_format="GLB", use_selection=False)
        print("exported", a.glb)


if __name__ == "__main__":
    main()
