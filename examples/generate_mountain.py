import os
import trimesh
import numpy as np
from typing import Tuple

from terrain_generator.trimesh_tiles.mesh_parts.mountain import generate_perlin_terrain
from terrain_generator.trimesh_tiles.mesh_parts.tree import add_trees_on_terrain


def generate_mountain(mesh_dir):

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

    os.makedirs(mesh_dir, exist_ok=True)
    mesh.export(os.path.join(mesh_dir, "mountain.obj"))
    terrain.export(os.path.join(mesh_dir, "terrain.obj"))
    tree_mesh.export(os.path.join(mesh_dir, "tree.obj"))


if __name__ == "__main__":
    mesh_dir = "results/mountains"
    generate_mountain(mesh_dir)
