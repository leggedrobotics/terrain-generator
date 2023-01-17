import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from wfc.wfc import WFCSolver

from mesh_parts.create_tiles import create_mesh_pattern
from mesh_parts.mesh_utils import visualize_mesh

from configs.each_part_cfg import IndoorPattern, IndoorPatternLevels
from alive_progress import alive_bar


def test_wall_mesh(
    mesh_name="result_mesh.stl",
    mesh_dir="results/result",
    shape=[20, 20],
    level_diff=0.5,
    level_n=5,
    wall_height=3.0,
    visualize=False,
    enable_history=False,
):

    dim = (2.0, 2.0, 2.0)
    levels = [np.round(level_diff * n, 2) for n in range(level_n)]
    print("levels = ", levels)
    cfg = IndoorPatternLevels(dim=dim, levels=tuple(levels), wall_height=wall_height)
    tiles = create_mesh_pattern(cfg)

    keywords = ["0.0_1.0"]
    exclude_keywords = ["90", "180", "270", "_x", "_y"]

    generate_tile_names = []

    for tile in tiles.values():
        # print(tile.name)
        if any([keyword in tile.name for keyword in keywords]):
            if not any([keyword in tile.name for keyword in exclude_keywords]):
                generate_tile_names.append(tile.name)
                print(tile.name)

    for init_name in generate_tile_names:
        generate_for_part(init_name, shape, tiles, mesh_dir, dim)

    # print(generate_tile_names)

    # init_name = "narrow_0.0_1.0_I"
    # init_name = "stepping_0.0_1.0_s"
    # init_name = "stair_0.0_1.0_wall_()_0"
    # init_name = "floating_boxes_0.0_1.0_29_x"


def generate_for_part(init_name, shape, tiles, mesh_dir, dim, visualize=False, enable_history=False):
    wfc_solver = WFCSolver(shape=shape, dimensions=2, seed=None)
    for tile in tiles.values():
        if visualize:
            print(tile.name)
        wfc_solver.register_tile(*tile.get_dict_tile())
    init_tiles = [
        # ("floor", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1.0_2.0_1111_f", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_0.0_1.0_1111_f", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        (init_name, (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
    ]
    wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)

    if enable_history:
        np.save(os.path.join(mesh_dir, "wave.npy"), wave)
        np.save(os.path.join(mesh_dir, "wave_order.npy"), wfc_solver.wfc.wave.wave_order)

    names = wfc_solver.names

    print("Converting to mesh...")
    result_mesh = trimesh.Trimesh()
    with alive_bar(len(wave.flatten())) as bar:
        for y in range(wave.shape[0]):
            for x in range(wave.shape[1]):
                mesh = tiles[names[wave[y, x]]].get_mesh().copy()
                if enable_history:
                    mesh.export(os.path.join(mesh_dir, f"{wave[y, x]}_{y}_{x}_{names[wave[y, x]]}.obj"))
                xy_offset = np.array([x * dim[0], -y * dim[1], 0.0])
                mesh.apply_translation(xy_offset)
                if enable_history:
                    mesh.export(os.path.join(mesh_dir, f"{wave[y, x]}_{y}_{x}_{names[wave[y, x]]}_translated.obj"))
                result_mesh += mesh
                bar()

    bbox = result_mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Get the size of the bounding box.
    result_mesh = result_mesh.apply_translation(-center)

    mesh_name = os.path.join(mesh_dir, f"{init_name}.obj")

    print("saving mesh to ", mesh_name)
    result_mesh.export(mesh_name)
    if visualize:
        visualize_mesh(result_mesh)


if __name__ == "__main__":

    # level_diffs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    # wall_heights = [0.15, 0.3, 1.0, 2.0, 3.0, 3.0]
    level_diffs = [0.5]
    wall_heights = [3.0]
    # level_diffs = [0.1]
    # wall_heights = [0.3]

    for level_diff, wall_height in zip(level_diffs, wall_heights):
        # result_dir = f"results/level_{level_diff}_floating_platform1"
        result_dir = f"results/test_each_parts"
        os.makedirs(result_dir, exist_ok=True)
        for i in range(1):
            name = os.path.join(result_dir, f"mesh_{i}.stl")
            test_wall_mesh(
                name, result_dir, shape=[3, 3], level_diff=level_diff, wall_height=wall_height, visualize=True
            )
