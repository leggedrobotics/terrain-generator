import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from wfc.wfc import WFCSolver

from mesh_parts.create_tiles import create_mesh_pattern
from mesh_parts.mesh_utils import visualize_mesh

from configs.indoor_cfg import IndoorPattern, IndoorPatternLevels
from alive_progress import alive_bar


def test_wall_mesh(mesh_name="result_mesh.stl", level_diff=0.5, level_n=5, wall_height=3.0, visualize=False):

    dim = (2.0, 2.0, 2.0)
    levels = [np.round(level_diff * n, 2) for n in range(level_n)]
    print("levels = ", levels)
    cfg = IndoorPatternLevels(dim=dim, levels=tuple(levels), wall_height=wall_height)
    tiles = create_mesh_pattern(cfg)

    wfc_solver = WFCSolver(shape=[24, 24], dimensions=2, seed=None)

    for tile in tiles.values():
        if visualize:
            print(tile)
        wfc_solver.register_tile(*tile.get_dict_tile())

    init_tiles = [
        ("floor", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_2_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
    ]
    # wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)

    try:
        wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)
    except Exception as e:
        print(e)
        return

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
    result_mesh = trimesh.Trimesh()
    with alive_bar(len(wave.flatten())) as bar:
        for y in range(wave.shape[0]):
            for x in range(wave.shape[1]):
                mesh = tiles[names[wave[y, x]]].get_mesh().copy()
                xy_offset = np.array([x * dim[0], -y * dim[1], 0.0])
                mesh.apply_translation(xy_offset)
                result_mesh += mesh
                bar()

    print("saving mesh to ", mesh_name)
    result_mesh.export(mesh_name)
    if visualize:
        visualize_mesh(result_mesh)


if __name__ == "__main__":

    level_diffs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    wall_heights = [0.15, 0.3, 1.0, 2.0, 3.0, 3.0]
    # level_diffs = [0.1]
    # wall_heights = [0.3]

    for level_diff, wall_height in zip(level_diffs, wall_heights):
        result_dir = f"results/level_{level_diff}"
        os.makedirs(result_dir, exist_ok=True)
        for i in range(10):
            name = os.path.join(result_dir, f"mesh_{i}.stl")
            test_wall_mesh(name, level_diff, wall_height=wall_height, visualize=False)
