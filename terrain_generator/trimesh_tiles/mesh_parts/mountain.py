#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import trimesh
import numpy as np
from typing import Tuple

from ...utils import convert_heightfield_to_trimesh
from .tree import add_trees_on_terrain
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d


def generate_perlin_terrain(
    base_shape: Tuple = (256, 256),
    base_res: Tuple = (2, 2),
    base_octaves: int = 2,
    base_fractal_weight: float = 0.2,
    noise_res: Tuple = (4, 4),
    noise_octaves: int = 5,
    base_scale: float = 2.0,
    noise_scale: float = 1.0,
    horizontal_scale: float = 1.0,
    vertical_scale: float = 10.0,
):
    # Generate fractal noise instead of Perlin noise
    base = generate_perlin_noise_2d(base_shape, base_res, tileable=(True, True))
    base += generate_fractal_noise_2d(base_shape, base_res, base_octaves, tileable=(True, True)) * base_fractal_weight

    # Use different weights for the base and noise heightfields
    noise = generate_fractal_noise_2d(base_shape, noise_res, noise_octaves, tileable=(True, True))

    terrain_height = base * base_scale + noise * noise_scale

    terrain_mesh = convert_heightfield_to_trimesh(terrain_height, horizontal_scale, vertical_scale)

    terrain_mesh.vertices[:, 2] = trimesh.smoothing.filter_humphrey(terrain_mesh).vertices[:, 2]

    return terrain_mesh


if __name__ == "__main__":
    terrain = generate_perlin_terrain(horizontal_scale=0.2, vertical_scale=3.0)
    # terrain.show()
    tree_mesh = add_trees_on_terrain(
        terrain, num_trees=100, tree_scale_range=(0.15, 0.55), tree_deg_range=(-60, 60), tree_cylinder_sections=4
    )
    mesh = terrain + tree_mesh
    # mesh.show()
    bbox = mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Translate the mesh to the center of the bounding box.
    mesh = mesh.apply_translation(-center)
    mesh.export("mountain.obj")
