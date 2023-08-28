#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
import trimesh
import networkx as nx
import open3d as o3d
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import cv2

from .mesh_utils import get_heights_from_mesh


def get_height_array_of_mesh_with_resolution(mesh, resolution=0.4, border_offset=0.0, return_points: bool = False):
    # Get the bounding box of the mesh.
    bbox = mesh.bounding_box.bounds
    # Get the minimum and maximum of the bounding box.
    b_min = np.min(bbox, axis=0)
    b_max = np.max(bbox, axis=0)
    center = (b_min + b_max) / 2

    dim = np.array([b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2]])

    n_points = int((b_max[0] - b_min[0]) / resolution)

    x = np.linspace(b_min[0] + border_offset, b_max[0] - border_offset, n_points)
    y = np.linspace(b_min[1] + border_offset, b_max[1] - border_offset, n_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)

    heights = get_heights_from_mesh(mesh, origins)
    array = heights.reshape(n_points, n_points)
    origins[:, 2] = heights
    if return_points:
        return array, center[:2], origins
    else:
        return array, center[:2]


def calc_spawnable_locations_on_terrain(
    mesh: trimesh.Trimesh,
    # num_points=1000,
    filter_size=(5, 5),
    spawnable_threshold=0.1,
    border_offset=1.0,
    resolution=0.4,
    # n_points_per_tile=5,
    visualize=False,
):
    """
    Create spawnable locations from a mesh.
    Args :param mesh: Mesh to create spawnable locations from.
    Returns: Spawnable locations.
    """
    array, center, origins = get_height_array_of_mesh_with_resolution(
        mesh, resolution=resolution, border_offset=border_offset, return_points=True
    )

    if visualize:
        plt.imshow(array)
        plt.colorbar()
        plt.show()

    flat_filter = np.ones(filter_size) * -1
    flat_filter[filter_size[0] // 2, filter_size[1] // 2] = flat_filter.shape[0] * flat_filter.shape[1] - 1

    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_img = cv2.filter2D(array, -1, flat_filter)

    if visualize:
        plt.imshow(filtered_img)
        plt.colorbar()
        plt.show()

        # visualize thresholded image
        plt.imshow(filtered_img < spawnable_threshold)
        plt.colorbar()
        plt.show()

    spawnable_indices = np.argwhere(np.abs(filtered_img) <= spawnable_threshold)
    # spawnable_idx = np.random.choice(np.arange(len(spawnable_indices)), num_points)
    # spawnable_idx = spawnable_indices[spawnable_idx]
    # Make it sparse
    spawnable_idx = spawnable_indices
    spawnable_locations = origins.reshape((filtered_img.shape[0], filtered_img.shape[1], 3))[
        spawnable_idx[:, 0], spawnable_idx[:, 1]
    ]
    spawnable_heights = array[spawnable_idx[:, 0], spawnable_idx[:, 1]]

    spawnable_locations[:, 2] = spawnable_heights

    return spawnable_locations


def get_sdf_of_points(points, sdf_array, center, resolution):
    """Deplicated"""
    point = points
    point = point - center
    point = point / resolution
    point = point.round().astype(int)
    point += np.array(sdf_array.shape) // 2
    is_valid = np.logical_and(
        np.logical_and(
            np.logical_and(point[:, 0] >= 0, point[:, 0] < sdf_array.shape[0]),
            np.logical_and(point[:, 1] >= 0, point[:, 1] < sdf_array.shape[1]),
        ),
        np.logical_and(point[:, 2] >= 0, point[:, 2] < sdf_array.shape[2]),
    )
    point[:, 0] = np.clip(point[:, 0], 0, sdf_array.shape[0] - 1)
    point[:, 1] = np.clip(point[:, 1], 0, sdf_array.shape[1] - 1)
    point[:, 2] = np.clip(point[:, 2], 0, sdf_array.shape[2] - 1)

    sdf = np.ones(point.shape[0]) * 1000.0
    sdf[is_valid] = sdf_array[point[is_valid, 0], point[is_valid, 1], point[is_valid, 2]]
    return sdf


def filter_spawnable_locations_with_sdf(
    spawnable_locations: np.ndarray,
    sdf_array: np.ndarray,
    height_offset: float = 0.5,
    sdf_resolution: float = 0.1,
    sdf_threshold: float = 0.2,
):
    # get sdf values for spawnable locations
    spawnable_locations[:, 2] += height_offset
    sdf_values = get_sdf_of_points(spawnable_locations, sdf_array, np.array([0, 0, 0]), sdf_resolution)
    # filter spawnable locations
    spawnable_locations = spawnable_locations[sdf_values > sdf_threshold]
    return spawnable_locations


def calc_spawnable_locations_with_sdf(
    terrain_mesh: trimesh.Trimesh,
    sdf_array: np.ndarray,
    visualize: bool = False,
    height_offset: float = 0.5,
    sdf_resolution: float = 0.1,
    sdf_threshold: float = 0.4,
):
    spawnable_locations = calc_spawnable_locations_on_terrain(terrain_mesh, visualize=visualize)
    spawnable_locations = filter_spawnable_locations_with_sdf(
        spawnable_locations, sdf_array, height_offset, sdf_resolution, sdf_threshold
    )
    return spawnable_locations


def locations_to_graph(positions):
    G = nx.Graph()

    # Add nodes to the graph with their positions
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)

    # Compute the distance matrix
    distances = np.sqrt(((positions[:, np.newaxis] - positions) ** 2).sum(axis=2))

    # Create a mask for edges below the threshold distance
    threshold = 0.5
    mask = distances < threshold

    # Create a list of edge tuples from the mask
    edges = np.transpose(np.where(mask))

    # Add edges to the graph
    G.add_edges_from(edges)

    # Print the graph nodes and edges
    # print("Nodes:", G.nodes(data=True))
    # print("Edges:", G.edges())
    return G


def visualize_mesh_and_graphs(
    mesh: trimesh.Trimesh,
    points: Union[nx.Graph, np.ndarray],
    color_values: Optional[np.ndarray] = None,
    goal_pos: Optional[np.ndarray] = None,
):

    if isinstance(points, nx.Graph):
        points = nx.get_node_attributes(points, "pos")
        points = np.array(list(points.values()))

    # print("points ", points, points.shape)

    # Create a point cloud where the points are the occupied SDF grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    viewer.add_geometry(pcd)

    # Mesh
    o3d_mesh = mesh.as_open3d
    o3d_mesh.compute_vertex_normals()

    viewer.add_geometry(o3d_mesh)

    # Set the point cloud colors based on the color values
    if color_values is not None:
        cmap = plt.get_cmap("rainbow")
        vmin = color_values.min()
        vmax = color_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sdf_colors = cmap(norm(color_values.flatten()))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(sdf_colors)

    if goal_pos is not None:
        goal_pos = goal_pos.reshape(-1)
        goal_pos = np.array([goal_pos[0], goal_pos[1], -0.5])
        line_points = [goal_pos, goal_pos + np.array([0, 0, 4])]
        lines = [[0, 1]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        viewer.add_geometry(line_set)

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_grid])

    # Visualize the point cloud and the mesh
    # viewer.add_geometry(voxel_grid)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    viewer.destroy_window()
    # o3d.visualization.draw_geometries([voxel_grid, o3d_mesh])


def create_2d_graph_from_height_array(
    height_array: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    invalid_cost=1000.0,
    use_diagonal: bool = True,
):
    # height_array = get_height_array_of_mesh_with_resolution(mesh, resolution=height_array_resolution)

    # graph_height_ratio = graph_resolution / height_array_resolution
    graph_shape = (np.array(height_array.shape) // graph_ratio).astype(int)

    G = nx.grid_2d_graph(*graph_shape)
    # Add diagonal edges
    if use_diagonal:
        G.add_edges_from(
            [((x, y), (x + 1, y + 1)) for x in range(graph_shape[0] - 1) for y in range(graph_shape[1] - 1)]
            + [((x + 1, y), (x, y + 1)) for x in range(graph_shape[0] - 1) for y in range(graph_shape[1] - 1)]
        )
    # Add cost map to edges
    for (u, v) in G.edges():
        cost = height_map_cost(u, v, height_array, graph_ratio, height_threshold, invalid_cost)
        G[u][v]["weight"] = cost
    return G


def distance_matrix_from_graph(graph: nx.Graph):
    # Compute adjacency matrix
    adj_mtx = nx.adjacency_matrix(graph)

    g_mat = csr_matrix(adj_mtx)

    # Compute shortest path distances using Dijkstra's algorithm
    dist_matrix, _ = shortest_path(csgraph=g_mat, directed=False, return_predecessors=True)
    return dist_matrix


def compute_distance_matrix(
    mesh: trimesh.Trimesh,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
    height_map_resolution: float = 0.1,
):
    height_array, center = get_height_array_of_mesh_with_resolution(mesh, resolution=height_map_resolution)
    G = create_2d_graph_from_height_array(
        height_array, graph_ratio=graph_ratio, invalid_cost=invalid_cost, height_threshold=height_threshold
    )
    dist_matrix = distance_matrix_from_graph(G)
    shape = (np.array(height_array.shape) // graph_ratio).astype(int)
    return dist_matrix, shape, center


def height_map_cost(
    u: Tuple[int, int],
    v: Tuple[int, int],
    height_array: np.ndarray,
    ratio: int,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
):
    # sample heights between u and v in height array.
    # number of points is determined by ratio
    ratio = int(ratio)
    um = np.array(u) * ratio
    vm = np.array(v) * ratio
    idx = np.linspace(um, vm, num=ratio + 1).astype(int)
    heights = height_array[idx[:, 0], idx[:, 1]]
    diffs = np.abs(heights[1:] - heights[:-1])
    distance = np.linalg.norm(vm - um) / ratio
    costs = distance + invalid_cost * (diffs > height_threshold)
    return costs.sum()


def visualize_distance(height_array, distance_matrix, graph_ratio, goal_pos, height_array_resolution=0.1):

    distance_shape = (np.array(height_array.shape) // graph_ratio).astype(int)
    grid_x, grid_y = np.meshgrid(np.arange(distance_shape[0]), np.arange(distance_shape[1]), indexing="ij")
    grid_z = height_array[grid_x * graph_ratio, grid_y * graph_ratio]
    distance_points = np.stack(
        [
            grid_x.flatten() * graph_ratio * height_array_resolution,
            grid_y.flatten() * graph_ratio * height_array_resolution,
            grid_z.flatten(),
        ],
        axis=1,
    )
    goal_idx = goal_pos[0] * distance_shape[0] + goal_pos[1]
    distances = distance_matrix[goal_idx, :]
    # Create a point cloud where the points are the occupied SDF grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(distance_points)

    # Set the point cloud colors based on the SDF values
    cmap = plt.get_cmap("rainbow")
    norm = plt.Normalize(vmin=0.0, vmax=150.0)
    distance_colors = cmap(norm(distances.flatten()))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(distance_colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.20)

    # Visualize the point cloud and the mesh
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(voxel_grid)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    viewer.destroy_window()
