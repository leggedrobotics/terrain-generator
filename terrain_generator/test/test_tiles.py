#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np

import sys

from ..wfc.tiles import Tile, ArrayTile, MeshTile


def test_tile():
    tile = Tile(name="tile1", edges={"up": "edge1", "right": "edge2", "down": "edge3", "left": "edge4"})
    print(tile)
    assert tile.name == "tile1"

    rotated_tile = tile.get_rotated_tile(90)
    print("rotated_tile", rotated_tile)

    flipped_tile = tile.get_flipped_tile("x")
    print("flipped", flipped_tile)

    tiles = tile.get_all_tiles(rotations=(90, 180, 270), flips=("x", "y"))
    for tile in tiles:
        print(tile)


def test_array_tile():
    tile = ArrayTile(name="tile", array=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    print(tile)

    rotated_tile = tile.get_rotated_tile(90)
    print("rotated_tile", rotated_tile)
    assert np.allclose(rotated_tile.array, np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]))
    assert rotated_tile.edges["up"] == tile.edges["right"]
    assert rotated_tile.edges["right"] == tile.edges["down"]
    assert rotated_tile.edges["down"] == tile.edges["left"]
    assert rotated_tile.edges["left"] == tile.edges["up"]

    rotated_tile = tile.get_rotated_tile(270)
    print("rotated_tile", rotated_tile)
    assert np.allclose(rotated_tile.array, np.array([[6, 3, 0], [7, 4, 1], [8, 5, 2]]))
    assert rotated_tile.edges["right"] == tile.edges["up"]
    assert rotated_tile.edges["down"] == tile.edges["right"]
    assert rotated_tile.edges["left"] == tile.edges["down"]
    assert rotated_tile.edges["up"] == tile.edges["left"]

    flipped_tile = tile.get_flipped_tile("x")
    print("flipped", flipped_tile)
    assert np.allclose(flipped_tile.array, np.array([[2, 1, 0], [5, 4, 3], [8, 7, 6]]))
    assert flipped_tile.edges["up"] == tile.edges["up"][::-1]
    assert flipped_tile.edges["right"] == tile.edges["left"][::-1]
    assert flipped_tile.edges["down"] == tile.edges["down"][::-1]
    assert flipped_tile.edges["left"] == tile.edges["right"][::-1]

    flipped_tile = tile.get_flipped_tile("y")
    print("flipped", flipped_tile)
    assert np.allclose(flipped_tile.array, np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]))
    assert flipped_tile.edges["up"] == tile.edges["down"][::-1]
    assert flipped_tile.edges["right"] == tile.edges["right"][::-1]
    assert flipped_tile.edges["down"] == tile.edges["up"][::-1]
    assert flipped_tile.edges["left"] == tile.edges["left"][::-1]
