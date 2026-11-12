"""Blender (4.2+/5.x) helper: decimate a mesh and Smart-UV-unwrap it, export OBJ with UVs.

  blender --background --python scripts/uv_unwrap.py -- --input export/room.ply --output export/room_uv.obj --target-tris 150000

Axes are exported unchanged (Z up, Y forward), so coordinates stay identical to the PLY (meters, floor at Z=0).
"""
import argparse
import math
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-tris", type=int, default=150_000)
    ap.add_argument("--angle", type=float, default=66.0, help="Smart UV angle limit (degrees)")
    ap.add_argument("--margin", type=float, default=0.002, help="island margin (UV units)")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=a.input)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    for attr in list(obj.data.color_attributes):  # colours are re-created from the photos
        obj.data.color_attributes.remove(attr)

    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if tris > a.target_tris:
        m = obj.modifiers.new("Decimate", "DECIMATE")
        m.ratio = a.target_tris / tris
        bpy.ops.object.modifier_apply(modifier=m.name)
    print("triangles:", tris, "->", sum(len(p.vertices) - 2 for p in obj.data.polygons))

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.uv.smart_project(angle_limit=math.radians(a.angle), island_margin=a.margin)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=a.output, export_selected_objects=True, export_uv=True, export_normals=False,
                          export_materials=False, forward_axis="Y", up_axis="Z")
    print("UV mesh written:", a.output)


if __name__ == "__main__":
    main()
