import numpy as np
import matplotlib.pyplot as plt
import trimesh

from wfc.wfc import WFCSolver
from wfc.tiles import Tile, ArrayTile

from create_indoor_mesh import create_mesh_pattern
from mesh_parts.mesh_parts_cfg import FloorPattern, StairsPattern

def test_wall_mesh():

    dim = (2.0, 2.0, 2.0)
    cfg = FloorPattern(name="floor", dim=dim)
    print("cfg ", cfg)
    tiles = create_mesh_pattern(cfg)

    cfg_s = StairsPattern(name="floor", dim=dim)
    print("cfg s", cfg_s)
    tiles.update(create_mesh_pattern(cfg_s))
    

    for tile in tiles.values():
        print(tile)

    wfc_solver = WFCSolver(shape=[10, 10], dimensions=2, seed=None)

    for tile in tiles.values():
        print("tile ", tile)
        wfc_solver.register_tile(*tile.get_dict_tile())

    # init_args = {"idx": (3, 3), "tile_name": "FA"}
    # wave = wfc_solver.run(init_args=init_args)
    wave = wfc_solver.run()
    # tile_arrays = {}
    # for tile in tiles.values():
    #     tile_arrays[tile.name] = tile.get_array()

    tile_array = tiles["floor"].get_array()
    array_shape = tile_array.shape
    img = np.zeros((wave.shape[0] * array_shape[0], wave.shape[1] * array_shape[1]))
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            # tile = tile_arrays[wfc_solver.names[wave[y, x]]]
            tile = tiles[wfc_solver.names[wave[y, x]]].get_array()
            img[y * array_shape[0] : (y + 1) * array_shape[0], x * array_shape[1] : (x + 1) * array_shape[1]] = tile

    plt.imshow(img)
    plt.colorbar()
    plt.show()

    names = wfc_solver.names


    result_mesh = trimesh.Trimesh()
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            # print("name ", names[wave[y, x]], x, y)
            # mesh = meshes.get_mesh(names[wave[y, x]])
            # print("mesh ", tiles[names[wave[y, x]]])
            mesh = tiles[names[wave[y, x]]].get_mesh().copy()
            # print("array ", tiles[names[wave[y, x]]].get_array())
            # mesh.show()
            xy_offset = np.array([x * dim[0], -y * dim[1], 0.0])
            mesh.apply_translation(xy_offset)
            result_mesh += mesh

    # result_mesh.export("result_mesh.stl")
    result_mesh.show()


if __name__ == "__main__":
    test_wall_mesh()
