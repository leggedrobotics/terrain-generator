import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from wfc.wfc import WFCSolver

from mesh_parts.create_tiles import create_mesh_pattern
from mesh_parts.mesh_utils import visualize_mesh

from configs.indoor_cfg import IndoorPattern, IndoorPatternLevels
from configs.navigation_cfg import IndoorNavigationPatternLevels
from configs.overhanging_cfg import OverhangingPattern
from alive_progress import alive_bar


def create_tiles(
    mesh_name="result_mesh.stl",
    mesh_dir="results/result",
    cfg_class=IndoorPatternLevels,
    shape=[20, 20],
    init_name="floor",
    level_diff=0.5,
    level_n=5,
    wall_height=3.0,
    visualize=False,
    enable_history=False,
):

    dim = (2.0, 2.0, 2.0)
    levels = [np.round(level_diff * n, 2) for n in range(level_n)]
    print("levels = ", levels)
    # cfg = IndoorPatternLevels(dim=dim, levels=tuple(levels), wall_height=wall_height)
    cfg = cfg_class(dim=dim)
    tiles = create_mesh_pattern(cfg)

    wfc_solver = WFCSolver(shape=shape, dimensions=2, seed=None)

    for tile in tiles.values():
        if visualize:
            print(tile.name)
        wfc_solver.register_tile(*tile.get_dict_tile())

    # init_name = "narrow_0.0_1.0_I"
    # init_name = "stepping_0.0_1.0_s"

    init_tiles = [
        # ("floor", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1.0_2.0_1111_f", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_0.0_1.0_1111_f", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        (init_name, (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
    ]
    wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)

    # try:
    #     wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)
    # except Exception as e:
    #     print(e)
    #     return

    # wave_history = []
    # is_collapsed_history = []
    # for w in wfc_solver.get_history():
    #     print(w.wave)
    #     print(w.is_collapsed)
    #     wave_history.append(w.wave)
    #     is_collapsed_history.append(w.is_collapsed)
    #
    # print("wave order ", wfc_solver.wfc.wave.wave_order)
    #
    # wave_history = np.array(wave_history)
    # is_collapsed_history = np.array(is_collapsed_history)
    # np.save(os.path.join(mesh_dir, "wave_history.npy"), wave_history)
    # np.save(os.path.join(mesh_dir, "is_collapsed_history.npy"), is_collapsed_history)
    # history = wfc_solver.get_history()
    # np.save(os.path.join(mesh_dir, "history.npy"), history)
    if enable_history:
        np.save(os.path.join(mesh_dir, "wave.npy"), wave)
        np.save(os.path.join(mesh_dir, "wave_order.npy"), wfc_solver.wfc.wave.wave_order)

    tile_array = tiles["floor"].get_array()
    array_shape = tile_array.shape
    img = np.zeros((wave.shape[0] * array_shape[0], wave.shape[1] * array_shape[1]))
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            tile = tiles[wfc_solver.names[wave[y, x]]].get_array()
            img[y * array_shape[0] : (y + 1) * array_shape[0], x * array_shape[1] : (x + 1) * array_shape[1]] = tile

    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()

    names = wfc_solver.names

    print("Converting to mesh...")
    if enable_history:
        parts_dir = os.path.join(mesh_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        translated_parts_dir = os.path.join(mesh_dir, "translated_parts")
        os.makedirs(translated_parts_dir, exist_ok=True)
    result_mesh = trimesh.Trimesh()
    with alive_bar(len(wave.flatten())) as bar:
        for y in range(wave.shape[0]):
            for x in range(wave.shape[1]):
                mesh = tiles[names[wave[y, x]]].get_mesh().copy()
                if enable_history:
                    mesh.export(os.path.join(parts_dir, f"{wave[y, x]}_{y}_{x}_{names[wave[y, x]]}.obj"))
                xy_offset = np.array([x * dim[0], -y * dim[1], 0.0])
                mesh.apply_translation(xy_offset)
                if enable_history:
                    mesh.export(
                        os.path.join(translated_parts_dir, f"{wave[y, x]}_{y}_{x}_{names[wave[y, x]]}_translated.obj")
                    )
                result_mesh += mesh
                bar()

    bbox = result_mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Get the size of the bounding box.
    result_mesh = result_mesh.apply_translation(-center)

    print("saving mesh to ", mesh_name)
    result_mesh.export(mesh_name)
    if visualize:
        visualize_mesh(result_mesh)


def create_nav_mesh():
    level_diffs = [0.6]
    wall_heights = [3.0]
    # level_diffs = [0.1]
    # wall_heights = [0.3]
    # cfg_class = IndoorNavigationPatternLevels
    cfg_class = OverhangingPattern
    # init_name = "platform_0.0_2.0_1111"
    init_name = "floor"

    for level_diff, wall_height in zip(level_diffs, wall_heights):
        result_dir = f"results/overhanging"
        # result_dir = f"results/test_each_parts"
        os.makedirs(result_dir, exist_ok=True)
        for i in range(1):
            name = os.path.join(result_dir, f"mesh_{i}.obj")
            create_tiles(
                name,
                result_dir,
                shape=[20, 20],
                level_diff=level_diff,
                init_name=init_name,
                cfg_class=cfg_class,
                wall_height=wall_height,
                visualize=True,
                enable_history=False,
            )


if __name__ == "__main__":

    create_nav_mesh()

    # level_diffs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    # wall_heights = [0.15, 0.3, 1.0, 2.0, 3.0, 3.0]
    # level_diffs = [0.5]
    # wall_heights = [3.0]
    # # level_diffs = [0.1]
    # # wall_heights = [0.3]
    #
    # for level_diff, wall_height in zip(level_diffs, wall_heights):
    #     result_dir = f"results/level_{level_diff}_with_history"
    #     # result_dir = f"results/test_each_parts"
    #     os.makedirs(result_dir, exist_ok=True)
    #     for i in range(1):
    #         name = os.path.join(result_dir, f"mesh_{i}.obj")
    #         test_wall_mesh(
    #             name,
    #             result_dir,
    #             shape=[20, 20],
    #             level_diff=level_diff,
    #             wall_height=wall_height,
    #             visualize=False,
    #             enable_history=True,
    #         )
