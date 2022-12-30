import numpy as np
import matplotlib.pyplot as plt

from wfc.wfc import WFCSolver
from wfc.tiles import Tile, ArrayTile


def test_simple_array():

    tiles = []
    tiles.append(ArrayTile(name="A", array=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), rotations=(90,)))
    tiles.append(ArrayTile(name="B", array=np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="C", array=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), rotations=()))
    tiles.append(ArrayTile(name="I", array=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), rotations=(90,)))
    tiles.append(ArrayTile(name="D", array=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), rotations=()))
    tiles.append(ArrayTile(name="E", array=np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="F", array=np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="G", array=np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="H", array=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))

    wfc_solver = WFCSolver(shape=[20, 20], dimensions=2, seed=1234)

    for tile in tiles:
        wfc_solver.register_tile(*tile.get_tile())

    wave = wfc_solver.run()

    rotations = [90, 180, 270]
    names = [tile.name for tile in tiles]

    tile_arrays = {}

    for tile in tiles:
        tile_arrays[tile.name] = tile.get_array()
        for r in rotations:
            name = f"{tile.name}_{r}"
            tile_arrays[name] = tile.get_array(name)

    img = np.zeros((wave.shape[0] * 3, wave.shape[1] * 3))
    print("img ", img.shape)
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            tile = tile_arrays[wfc_solver.names[wave[y, x]]]
            img[y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3] = tile

    plt.imshow(img)
    plt.show()


def test_wall_array():

    tiles = []
    # 0: empty, 1: floor, 2: wall
    tiles.append(ArrayTile(name="F", array=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), rotations=()))
    tiles.append(ArrayTile(name="FA", array=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), rotations=()))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="FB", array=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="FC", array=np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="FD", array=np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="FE", array=np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="FF", array=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))

    # tiles.append(ArrayTile(name="WA", array=np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="WB", array=np.array([[2, 2, 2], [1, 1, 2], [1, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="WC", array=np.array([[2, 1, 2], [2, 1, 2], [2, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="WD", array=np.array([[1, 1, 1], [2, 1, 2], [2, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="WE", array=np.array([[1, 1, 1], [1, 1, 2], [2, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="WF", array=np.array([[2, 1, 2], [1, 1, 1], [2, 1, 2]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="WG", array=np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="WH", array=np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))

    tiles.append(ArrayTile(name="WI", array=np.array([[1, 1, 1], [2, 2, 1], [1, 2, 1]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))

    tiles.append(ArrayTile(name="EA", array=np.array([[0, 2, 2], [0, 2, 2], [0, 2, 2]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    tiles.append(ArrayTile(name="EB", array=np.array([[2, 2, 2], [1, 1, 2], [0, 1, 2]]), rotations=(90, 180, 270)))
    tiles.append(tiles[-1].get_flipped_tile("x"))
    tiles.append(tiles[-2].get_flipped_tile("y"))
    # tiles.append(ArrayTile(name="EC", array=np.array([[0, 0, 0], [1, 1, 0], [2, 1, 0]]), rotations=(90, 180, 270)))
    # tiles.append(tiles[-1].get_flipped_tile("x"))
    # tiles.append(tiles[-2].get_flipped_tile("y"))

    wfc_solver = WFCSolver(shape=[10, 10], dimensions=2, seed=None)

    for tile in tiles:
        wfc_solver.register_tile(*tile.get_tile())

    wave = wfc_solver.run()

    rotations = [90, 180, 270]
    names = [tile.name for tile in tiles]

    tile_arrays = {}

    for tile in tiles:
        tile_arrays[tile.name] = tile.get_array()
        for r in rotations:
            name = f"{tile.name}_{r}"
            tile_arrays[name] = tile.get_array(name)

    img = np.zeros((wave.shape[0] * 3, wave.shape[1] * 3))
    print("img ", img.shape)
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            tile = tile_arrays[wfc_solver.names[wave[y, x]]]
            img[y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3] = tile

    plt.imshow(img)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    test_wall_array()
