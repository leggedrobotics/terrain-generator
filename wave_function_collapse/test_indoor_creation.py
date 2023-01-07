import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from wfc.wfc import WFCSolver

from mesh_parts.create_tiles import create_mesh_pattern
from mesh_parts.mesh_utils import visualize_mesh

from configs.indoor_cfg import IndoorPattern
from alive_progress import alive_bar


def test_wall_mesh(mesh_name="result_mesh.stl", visualize=False):

    dim = (2.0, 2.0, 2.0)
    cfg = IndoorPattern(dim=dim)
    tiles = create_mesh_pattern(cfg)

    wfc_solver = WFCSolver(shape=[24, 24], dimensions=2, seed=None)

    for tile in tiles.values():
        if visualize:
            print(tile)
        wfc_solver.register_tile(*tile.get_dict_tile())

    init_tiles = [
        # ("floor", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        ("platform_2_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1_1111", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("platform_1111_f", (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2)),
        # ("floor", [wfc_solver.shape[0] // 4, wfc_solver.shape[1] // 4]),
        # ("floor", [wfc_solver.shape[0] // 4, 3 * wfc_solver.shape[1] // 4]),
        # ("platform_1111_f", [3 * wfc_solver.shape[0] // 4, wfc_solver.shape[1] // 4]),
        # ("platform_2222_f", [3 * wfc_solver.shape[0] // 4, 3 * wfc_solver.shape[1] // 4]),
    ]
    wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)

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
    result_dir = "results/results_4"
    os.makedirs(result_dir, exist_ok=True)
    for i in range(10):
        name = os.path.join(result_dir, f"result_mesh_{i}.stl")
        test_wall_mesh(name, visualize=False)
