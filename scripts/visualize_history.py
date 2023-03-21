import os
import glob
import numpy as np
import trimesh
import time

import open3d as o3d


def visualize_history(result_dir: str):
    # history = np.load(os.path.join(result_dir, "history.npy"), allow_pickle=True)
    wave = np.load(os.path.join(result_dir, "wave.npy"), allow_pickle=True)
    wave_order = np.load(os.path.join(result_dir, "wave_order.npy"), allow_pickle=True)
    print("wave_order ", wave_order)
    # print(history[-1].wave)

    # tile_history = []
    # prev_is_collapsed = history[0].is_collapsed
    # for h in history:
    #     # print(h.wave)
    #     # print(h.is_collapsed)
    #     changed = h.is_collapsed != prev_is_collapsed
    #     # print("changed ", changed)
    #     changed_idx = np.array(changed.nonzero())
    #     if changed_idx.sum() > 0:
    #         changed_idx = np.array(changed_idx).T[0]
    #         # print("changed_idx ", changed_idx)
    #         # print("wave ", h.wave[changed_idx[0], changed_idx[1]])
    #         tile_history.append([h.wave[changed_idx[0], changed_idx[1]], changed_idx])
    #         prev_is_collapsed = h.is_collapsed

    # print(tile_history)

    meshes = {}
    # all mesh files
    for file in glob.glob(os.path.join(result_dir, "*.obj")):
        # print(file)
        # print(os.path.basename(file))
        meshes[os.path.basename(file)] = trimesh.load(file)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # print("tile_history ", len(tile_history))
    # exit(0)
    # mesh = trimesh.Trimesh()
    # for tile in tile_history:
    for i in range(wave_order.shape[0] * wave_order.shape[1]):
        mesh_idx = np.array(np.where(wave_order == i)).T[0]
        print("idx ", mesh_idx)
        mesh_id = wave[mesh_idx[0], mesh_idx[1]]
        # mesh_id = tile[0]
        # mesh_idx = tile[1]
        print("mesh_id ", mesh_id)
        # print("mesh_idx ", mesh_idx)
        keywords = ["translated", f"{mesh_id}_{mesh_idx[0]}_{mesh_idx[1]}_"]
        for key in meshes.keys():
            if all(x in key for x in keywords):
                print("found ", key)
                # mesh += meshes[key]
                mesh = meshes[key]

        mesh.export(os.path.join(result_dir, f"mesh_part_{i}.obj"))

        o3d_mesh = mesh.as_open3d
        R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
        print("R ", R)
        o3d_mesh.rotate(R, center=[0, 0, 0])
        o3d_mesh.compute_vertex_normals()
        # vis.clear_geometries()
        vis.add_geometry(o3d_mesh)
        vis.poll_events()
        view_control = vis.get_view_control()
        view_control.set_zoom(1.5)
        # view_control.camera_local_translate(vis, forward=-10.0, up=-0.5)
        # view_control.set_lookat([0, 0, 0])

        time.sleep(0.05)
        # vis.
    time.sleep(1000)


visualize_history("results/test_history_2")
