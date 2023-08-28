#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
import glob
import numpy as np
import trimesh
import time

import open3d as o3d
import argparse


def visualize_history(
    result_dir: str,
    rotate_x: float = 0.0,
    rotate_y: float = 0.0,
    rotate_z: float = 0.0,
    zoom: float = 1.5,
    sleep_time: float = 0.05,
    export_meshes_in_order: bool = False,
    output_dir: str = "mesh_parts",
    output_prefix: str = "mesh_part",
):
    wave = np.load(os.path.join(result_dir, "wave.npy"), allow_pickle=True)
    wave_order = np.load(os.path.join(result_dir, "wave_order.npy"), allow_pickle=True)
    print("wave_order ", wave_order)

    meshes = {}
    # all mesh files
    for file in glob.glob(os.path.join(result_dir, "translated_parts/*.obj")):
        meshes[os.path.basename(file)] = trimesh.load(file)

    if export_meshes_in_order:
        output_dir = os.path.join(result_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(wave_order.shape[0] * wave_order.shape[1]):
        mesh_idx = np.array(np.where(wave_order == i)).T[0]
        mesh_id = wave[mesh_idx[0], mesh_idx[1]]
        keywords = ["translated", f"{mesh_id}_{mesh_idx[0]}_{mesh_idx[1]}_"]
        for key in meshes.keys():
            if all(x in key for x in keywords):
                mesh = meshes[key]

        if export_meshes_in_order:
            mesh.export(os.path.join(output_dir, f"{output_prefix}_{i:05d}.obj"))

        o3d_mesh = mesh.as_open3d
        R = o3d.geometry.get_rotation_matrix_from_xyz([rotate_x, rotate_y, rotate_z])
        o3d_mesh.rotate(R, center=[0, 0, 0])
        o3d_mesh.compute_vertex_normals()
        vis.add_geometry(o3d_mesh)
        vis.poll_events()
        view_control = vis.get_view_control()
        view_control.set_zoom(zoom)

        time.sleep(sleep_time)
    time.sleep(1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize mesh history")
    parser.add_argument(
        "--result_dir", type=str, default="results/test_history_2", help="Directory containing the mesh history files"
    )
    parser.add_argument("--rotate_x", type=float, default=-1.0, help="Amount to rotate the mesh around the X axis")
    parser.add_argument("--rotate_y", type=float, default=0.0, help="Amount to rotate the mesh around the Y axis")
    parser.add_argument("--rotate_z", type=float, default=0.2, help="Amount to rotate the mesh around the Z axis")
    parser.add_argument("--zoom", type=float, default=1.5, help=" -- Initial zoom level for the visualization")
    parser.add_argument(
        "--sleep_time", type=float, default=0.05, help="Amount of time to sleep between each frame during visualization"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mesh_parts",
        help="Directory to save the generated mesh parts during visualization",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="mesh_part",
        help="Prefix to use for the generated mesh parts during visualization",
    )
    parser.add_argument(
        "--save", action="store_true", help="enable saving of the generated mesh parts to the output directory"
    )
    args = parser.parse_args()
    visualize_history(
        args.result_dir,
        args.rotate_x,
        args.rotate_y,
        args.rotate_z,
        args.zoom,
        args.sleep_time,
        args.save,
        args.output_dir,
        args.output_prefix,
    )
