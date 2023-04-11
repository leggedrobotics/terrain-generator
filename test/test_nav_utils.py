import numpy as np
import trimesh

from trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile, get_mesh_gen
from trimesh_tiles.mesh_parts.mesh_parts_cfg import FloatingBoxesPartsCfg, WallMeshPartsCfg, PlatformMeshPartsCfg
from trimesh_tiles.mesh_parts.overhanging_parts import create_overhanging_boxes, generate_wall_from_array

from utils import (
    calc_spawnable_locations_on_terrain,
    locations_to_graph,
    visualize_mesh_and_graphs,
    filter_spawnable_locations_with_sdf,
    get_height_array_of_mesh_with_resolution,
    create_2d_graph_from_height_array,
)
from utils.mesh_utils import get_height_array_of_mesh


def test_spawnable_location(visualize):
    # Load SDF array
    sdf_array = np.load("results/overhanging_with_sdf_no_wall/mesh_0.obj.npy")
    terrain_mesh = trimesh.load("results/overhanging_with_sdf_no_wall/mesh_0.obj_terrain.obj")
    all_mesh = trimesh.load("results/overhanging_with_sdf_no_wall/mesh_0.obj")
    spawnable_locations = calc_spawnable_locations_on_terrain(
        terrain_mesh, filter_size=(5, 5), visualize=False, n_points_per_tile=10
    )
    if visualize:
        print("spawnable_locations", spawnable_locations, spawnable_locations.shape)
        new_spawnable_locations = filter_spawnable_locations_with_sdf(
            spawnable_locations, sdf_array, height_offset=0.5, sdf_resolution=0.1, sdf_threshold=0.4
        )

        # sampled_points = spawnable_locations[np.random.choice(spawnable_locations.shape[0], 1000)]
        print("new spawnable_locations", new_spawnable_locations, new_spawnable_locations.shape)
        # graph = locations_to_graph(new_spawnable_locations)
        visualize_mesh_and_graphs(all_mesh, new_spawnable_locations)


def test_create_2d_graph_from_height_array(visualize, load_mesh=True):
    # import numpy as np
    import networkx as nx
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    import matplotlib.pyplot as plt

    if load_mesh:
        mesh = trimesh.load("results/overhanging_with_sdf_no_wall/mesh_0.obj_terrain.obj")
        height_array = get_height_array_of_mesh_with_resolution(mesh, resolution=0.1)
        plt.imshow(height_array)
        plt.show()
    else:
        height_array = np.zeros((80, 80))
        height_array[20, :60] = 2.0
        height_array[40, 20:] = 2.0
        height_array[60, :50] = 2.0

    G = create_2d_graph_from_height_array(height_array, graph_ratio=4)

    if visualize:
        grid_size = np.array(height_array.shape) // 4

        # Compute adjacency matrix
        adj_mtx = nx.adjacency_matrix(G)
        print("adj_mtx", adj_mtx, adj_mtx.shape)

        graph = csr_matrix(adj_mtx)

        # Compute shortest path distances using Dijkstra's algorith
        dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)

        # dist_mtx, _ = dijkstra(csgraph=adj_mtx, directed=False, return_predecessors=False)

        # Print results
        print("Distance matrix:")
        print(dist_matrix, dist_matrix.shape)
        print("d(10, 25) = ", dist_matrix[10, 25])
        nodes_list = np.array(list(G.nodes()))
        print("node (10, 25) = ", nodes_list[10], nodes_list[100])
        dist_from_10 = dist_matrix[5000, :]
        print("dist_from_10", dist_from_10.shape)
        img = dist_from_10.reshape(grid_size)
        print("img", img.shape)
        plt.imshow(img, vmax=1000)
        # maximum value in the img is 300
        plt.show()
