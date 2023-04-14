import numpy as np
import torch
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
    distance_matrix_from_graph,
    create_2d_graph_from_height_array,
    visualize_distance,
)
from utils.mesh_utils import get_height_array_of_mesh

from navigation.mesh_terrain import MeshTerrain, MeshTerrainCfg, SDFArray, NavDistance


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

        np.save("results/overhanging_with_sdf_no_wall/dist_matrix.npy", dist_matrix)
        np.save("results/overhanging_with_sdf_no_wall/height_map.npy", height_array)

        # dist_mtx, _ = dijkstra(csgraph=adj_mtx, directed=False, return_predecessors=False)

        # Print results
        print("Distance matrix:")
        print(dist_matrix, dist_matrix.shape)
        print("d(10, 25) = ", dist_matrix[10, 25])
        nodes_list = np.array(list(G.nodes()))
        print("node (10, 25) = ", nodes_list[10], nodes_list[100])
        dist_from_10 = dist_matrix[50, :]
        print("dist_from_10", dist_from_10.shape)
        img = dist_from_10.reshape(grid_size)
        print("img", img.shape)
        plt.imshow(img, vmax=100)
        # maximum value in the img is 300
        plt.show()
        visualize_distance(height_array, dist_matrix, 4, (60, 30), height_array_resolution=0.1)


def test_visualize_distance(visualize):
    mesh = trimesh.load("results/overhanging_with_sdf_no_wall/mesh_0.obj_terrain.obj")
    height_array = np.load("results/overhanging_with_sdf_no_wall/height_map.npy")
    dist_matrix = np.load("results/overhanging_with_sdf_no_wall/dist_matrix.npy")
    ratio = 4
    visualize_distance(height_array, dist_matrix, ratio, (20, 80), height_array_resolution=0.1)


def test_nav_mesh(visualize):
    cfg = MeshTerrainCfg(
        mesh_path="results/overhanging_with_sdf_no_wall/mesh_0.obj_terrain.obj",
        distance_path="results/overhanging_with_sdf_no_wall/dist_matrix.npy",
        sdf_path="results/overhanging_with_sdf_no_wall/mesh_0.obj.npy",
    )
    print("cfg", cfg)
    mesh_terrain = MeshTerrain(cfg)
    print("mesh_terrain", mesh_terrain)


def test_sdf_array(visualize):
    sdf_array = torch.Tensor([[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]]).reshape(3, 3, 1)
    sdf_array = SDFArray(sdf_array, resolution=0.1)

    points = torch.Tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.05, 0.05, 0.00]])
    sdf = sdf_array.get_sdf(points)
    if visualize:
        print("sdf_array", sdf_array)
        print("sdf", sdf)

    expected_values = torch.Tensor([5.0, 9.0, 7.0])
    assert torch.allclose(sdf, expected_values)


def test_nav_distance(visualize):
    height_array = np.zeros((80, 80))
    height_array[20, :60] = 2.0
    height_array[40, 20:] = 2.0
    height_array[60, :50] = 2.0
    G = create_2d_graph_from_height_array(height_array, graph_ratio=4, invalid_cost=1000)
    dist_matrix = distance_matrix_from_graph(G)

    nav_distance = NavDistance(dist_matrix, shape=(20, 20), resolution=0.4)

    # points = torch.Tensor([[0.0, 0.0], [0.1, 0.1], [0.05, 0.05]])
    # get points from mesh grid
    x = np.linspace(-3.99, 3.99, 200)
    y = np.linspace(-3.99, 3.99, 200)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv, yv], axis=2).reshape(-1, 2)
    goal_pos = torch.Tensor([0.0, -1.5])
    d = nav_distance.get_distance(points, goal_pos)
    if visualize:
        import matplotlib.pyplot as plt

        print("nav_distance", nav_distance)

        # first visualize the distance map
        distance_map = dist_matrix[120, :].reshape((20, 20))
        plt.imshow(distance_map.T, vmax=300)
        plt.show()

        print("points", points, points.shape)
        print("d", d, d.shape)
        img = d.reshape((200, 200))
        plt.imshow(img, vmax=300)
        plt.show()

    # expected_values = torch.Tensor([5.0, 9.0, 7.0])
    # assert torch.allclose(distances, expected_values)
