import numpy as np
import trimesh
import networkx as nx
import open3d as o3d
from typing import Union, Tuple

import matplotlib.pyplot as plt

from utils import get_height_array_of_mesh, get_heights_from_mesh
import cv2


def get_height_array_of_mesh_with_resolution(mesh, resolution=0.4, border_offset=0.0):
    # Get the bounding box of the mesh.
    bbox = mesh.bounding_box.bounds
    # Get the minimum and maximum of the bounding box.
    b_min = np.min(bbox, axis=0)
    b_max = np.max(bbox, axis=0)

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
    return array


def calc_spawnable_locations_on_terrain(
    mesh: trimesh.Trimesh,
    # num_points=1000,
    filter_size=(5, 5),
    spawnable_threshold=0.1,
    # border_offset=1.0,
    resolution=0.4,
    # n_points_per_tile=5,
    visualize=False,
):
    """
    Create spawnable locations from a mesh.
    Args :param mesh: Mesh to create spawnable locations from.
    Returns: Spawnable locations.
    """
    # Get the bounding box of the mesh.
    # bbox = mesh.bounding_box.bounds
    # # Get the minimum and maximum of the bounding box.
    # b_min = np.min(bbox, axis=0)
    # b_max = np.max(bbox, axis=0)
    #
    # dim = np.array([b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2]])
    #
    # n_points = int((b_max[0] - b_min[0]) * n_points_per_tile)
    #
    # x = np.linspace(b_min[0] + border_offset, b_max[0] - border_offset, n_points)
    # y = np.linspace(b_min[1] + border_offset, b_max[1] - border_offset, n_points)
    # xv, yv = np.meshgrid(x, y)
    # xv = xv.flatten()
    # yv = yv.flatten()
    # origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    #
    # heights = get_heights_from_mesh(mesh, origins)
    # array = heights.reshape(n_points, n_points)
    array = get_height_array_of_mesh_with_resolution(mesh, resolution=resolution, border_offset=border_offset)

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

    print("is valid", is_valid)

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


def visualize_mesh_and_graphs(mesh: trimesh.Trimesh, points: Union[nx.Graph, np.ndarray]):

    if isinstance(points, nx.Graph):
        points = nx.get_node_attributes(points, "pos")
        points = np.array(list(points.values()))

    print("points ", points, points.shape)

    # Create a point cloud where the points are the occupied SDF grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Set the point cloud colors based on the SDF values
    # cmap = plt.get_cmap("rainbow")
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # sdf_colors = cmap(norm(sdf_values.flatten()))[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(sdf_colors)

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_grid])

    # Visualize the point cloud and the mesh
    o3d_mesh = mesh.as_open3d
    o3d_mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    # viewer.add_geometry(voxel_grid)
    viewer.add_geometry(pcd)
    viewer.add_geometry(o3d_mesh)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    viewer.destroy_window()
    # o3d.visualization.draw_geometries([voxel_grid, o3d_mesh])


def create_2d_graph_from_height_array(height_array: np.ndarray, graph_ratio: int = 4, height_threshold: float = 0.4):
    # height_array = get_height_array_of_mesh_with_resolution(mesh, resolution=height_array_resolution)

    # graph_height_ratio = graph_resolution / height_array_resolution
    graph_shape = (np.array(height_array.shape) // graph_ratio).astype(int)
    print("graph shape", graph_shape)

    G = nx.grid_2d_graph(*graph_shape)
    print("G ", G)
    print("G.nodes():", G.nodes())
    # Add cost map to edges
    for (u, v) in G.edges():
        print("u, v: ", u, v)
        # cost = cost_map[u[0], u[1]] + cost_map[v[0], v[1]]
        cost = height_map_cost(u, v, height_array, graph_ratio, height_threshold)
        print("cost: ", cost)
        G[u][v]["weight"] = cost
    return G


def height_map_cost(
    u: Tuple[int, int],
    v: Tuple[int, int],
    height_array: np.ndarray,
    ratio: int,
    height_threshold: float = 0.4,
    invalid_cost: float = 10000.0,
):
    # sample heights between u and v in height array.
    # number of points is determined by ratio
    ratio = int(ratio)
    um = np.array(u) * ratio
    vm = np.array(v) * ratio
    idx = np.linspace(um, vm, num=ratio + 1).astype(int)
    print("idx ", idx)
    heights = height_array[idx[:, 0], idx[:, 1]]
    print("heights: ", heights)
    diffs = np.abs(heights[1:] - heights[:-1])
    # costs = diffs + invalid_cost * (diffs > height_threshold)
    costs = 1.0 + invalid_cost * (diffs > height_threshold)
    print("costs: ", costs)
    return costs.sum()
