import trimesh
import numpy as np
from typing import Tuple

from utils import convert_heightfield_to_trimesh
from trimesh_tiles.mesh_parts.tree import add_trees_on_terrain
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
    terrain = generate_perlin_terrain()
    tree_mesh = add_trees_on_terrain(terrain, num_trees=150, tree_scale_range=(0.2, 1.5), tree_cylinder_sections=4)
    mesh = terrain + tree_mesh
    mesh.show()
    mesh.export("mountain.obj")
