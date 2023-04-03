import numpy as np
import trimesh

from trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile, get_mesh_gen
from trimesh_tiles.mesh_parts.mesh_parts_cfg import FloatingBoxesPartsCfg, WallMeshPartsCfg, PlatformMeshPartsCfg
from trimesh_tiles.mesh_parts.overhanging_parts import create_overhanging_boxes, generate_wall_from_array

from utils import (
    create_spawnable_locations_on_terrain,
    locations_to_graph,
    visualize_mesh_and_graphs,
    filter_spawnable_locations_with_sdf,
)


def test_spawnable_location(visualize):
    # Load SDF array
    sdf_array = np.load("results/overhanging_with_sdf/mesh_0.obj.npy")
    terrain_mesh = trimesh.load("results/overhanging_with_sdf/mesh_0.obj_terrain.obj")
    all_mesh = trimesh.load("results/overhanging_with_sdf/mesh_0.obj")
    spawnable_locations = create_spawnable_locations_on_terrain(terrain_mesh, filter_size=(5, 5), visualize=visualize)
    if visualize:
        print("spawnable_locations", spawnable_locations, spawnable_locations.shape)
        new_spawnable_locations = filter_spawnable_locations_with_sdf(
            spawnable_locations, sdf_array, height_offset=0.5, sdf_resolution=0.1, sdf_threshold=0.5
        )
        print("new spawnable_locations", new_spawnable_locations, new_spawnable_locations.shape)
        graph = locations_to_graph(new_spawnable_locations)
        visualize_mesh_and_graphs(all_mesh, graph)
