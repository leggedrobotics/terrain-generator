import os
import numpy as np
from pyglet.window.key import P
import trimesh
from typing import Callable, Any, Optional, Union, Tuple, List, Literal, Iterable
from dataclasses import asdict, is_dataclass
import open3d as o3d
import copy

from trimesh.exchange import xyz


def merge_meshes(
    meshes: List[trimesh.Trimesh], minimal_triangles: bool = False, engine: str = "blender"
) -> trimesh.Trimesh:
    if minimal_triangles:
        mesh = trimesh.boolean.union(meshes, engine=engine)
    else:
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def flip_mesh(mesh: trimesh.Trimesh, direction: Literal["x", "y"]):
    """Flip a mesh in a given direction."""
    new_mesh = mesh.copy()
    if direction == "x":
        # Create the transformation matrix for inverting the mesh in the x-axis
        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [1, 0, 0])
    elif direction == "y":
        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 1, 0])
    else:
        raise ValueError(f"Direction {direction} is not defined.")
    # Apply the transformation to the mesh
    new_mesh.apply_transform(transform)
    return new_mesh


def rotate_mesh(mesh: trimesh.Trimesh, deg: Literal[90, 180, 270]):
    """Rotate a mesh in a given degree."""
    new_mesh = mesh.copy()
    if deg == 90:
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    elif deg == 180:
        transform = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    elif deg == 270:
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
    else:
        raise ValueError(f"Rotation degree {deg} is not defined.")
    new_mesh.apply_transform(transform)
    return new_mesh


def get_heights_from_mesh(mesh: trimesh.Trimesh, origins: np.ndarray):
    """Get the height of a mesh at a given origin.
    Args:
        mesh (trimesh.Trimesh): mesh
        origins (np.ndarray): origins
    Returns:
        heights (np.ndarray): heights
    """

    # if origins dimension is 2, add 3rd dimension
    if origins.shape[1] == 2:
        origins = np.concatenate([origins, np.ones((origins.shape[0], 1)) * 100], axis=1)

    vectors = np.stack(
        [np.zeros_like(origins[:, 0]), np.zeros_like(origins[:, 1]), -np.ones_like(origins[:, 2])], axis=-1
    )
    # # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    array = np.zeros((origins.shape[0]))
    if len(points) > 0:
        array[index_ray] = points[:, 2]
    return array


def get_height_array_of_mesh(
    mesh: trimesh.Trimesh, dim: Tuple[float, float, float], num_points: int, offset: float = 0.01
):
    """
    Get the height array of a mesh.
    Args:
        mesh: The mesh.
        dim: The dimension of the mesh.
        num_points: The number of points in the height array.
        offset: The offset of the height array.
    Returns:
        The height array.
    """
    # intersects_location requires origins to be the same shape as vectors
    array = np.zeros((num_points * num_points))
    if mesh.is_empty:
        return array.reshape((num_points, num_points))
    x = np.linspace(-dim[0] / 2.0 + offset, dim[0] / 2.0 - offset, num_points)
    y = np.linspace(dim[1] / 2.0 - offset, -dim[1] / 2.0 + offset, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    vectors = np.stack([np.zeros_like(xv), np.zeros_like(yv), -np.ones_like(xv)], axis=-1)
    # # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    if len(points) > 0:
        array[index_ray] = points[:, 2] + dim[2] / 2.0
        array = np.round(array, 1)
    array = array.reshape(num_points, num_points)
    return array


def convert_heightfield_to_trimesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: Optional[float] = None,
):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y_min = -num_cols * horizontal_scale / 2.0
    y_max = num_cols * horizontal_scale / 2.0

    x_min = -num_rows * horizontal_scale / 2.0
    x_max = num_rows * horizontal_scale / 2.0

    y = np.linspace(y_min, y_max, num_cols)
    x = np.linspace(x_min, x_max, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return mesh


def merge_two_height_meshes(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh):
    """
    Merge two heightfield meshes into one.
    Parameters:
        mesh1 (trimesh.Trimesh): first heightfield mesh
        mesh2 (trimesh.Trimesh): second heightfield mesh
    Returns:
        merged_mesh (trimesh.Trimesh): merged heightfield mesh
    """
    # merge the two meshes
    merged_mesh = trimesh.util.concatenate(mesh1, mesh2)

    # connect edges
    vertices = merged_mesh.vertices
    faces = merged_mesh.faces

    n_vertices = vertices.shape[0] // 2
    n = int(np.sqrt(n_vertices))
    indices = np.arange(n_vertices).reshape(n, n)

    new_faces = []
    edges = [indices[0, :], indices[-1, :], indices[:, 0], indices[:, -1]]
    for edge_1 in edges:
        ind0 = edge_1[:-1]
        ind1 = edge_1[1:]
        ind2 = ind0 + n_vertices
        ind3 = ind1 + n_vertices
        f1 = np.zeros((n - 1, 3), dtype=np.uint32)
        f1[:, 0] = ind0
        f1[:, 1] = ind1
        f1[:, 2] = ind2
        new_faces.append(f1)
        f2 = np.zeros((n - 1, 3), dtype=np.uint32)
        f2[:, 0] = ind2
        f2[:, 1] = ind3
        f2[:, 2] = ind1
        new_faces.append(f2)
    new_faces = np.vstack(new_faces)
    merged_mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack([faces, new_faces]))
    return merged_mesh


def visualize_mesh(mesh: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh], save_path=None):
    """Visualize a mesh."""
    # o3d_mesh = o3d.geometry.TriangleMesh()
    if isinstance(mesh, trimesh.Trimesh):
        o3d_mesh = mesh.as_open3d
        # mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        # mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d_mesh = mesh
    # Visualize meshes one by one with Open3D
    o3d_mesh.compute_vertex_normals()
    R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
    o3d_mesh.rotate(R, center=[0, 0, 0])
    o3d.visualization.draw_geometries([o3d_mesh])
    vis = o3d.visualization.Visualizer()
    vis.capture_screen_image(save_path)


def compute_signed_distance_and_closest_geometry(scene: o3d.t.geometry.RaycastingScene, query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points["points"].numpy(), axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points["geometry_ids"].numpy()


def compute_sdf(mesh: trimesh.Trimesh, dim=[2, 2, 2], resolution: float = 0.1) -> np.ndarray:

    # To prevent weird behavior when two surfaces are exactly at the same positions
    mesh.vertices += np.random.uniform(-1e-4, 1e-4, size=mesh.vertices.shape)
    # Convert mesh to Open3D TriangleMesh
    mesh_o3d = mesh.as_open3d
    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)

    # Create RaycastingScene and add mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_o3d)

    # Compute grid coordinates and query points
    bbox = mesh.bounds
    dim = np.array(dim)
    num_elements = np.ceil(np.array(dim) / 0.1).astype(int)
    x_min = -dim[0] / 2
    x_max = dim[0] / 2
    y_min = -dim[1] / 2
    y_max = dim[1] / 2
    z_min = -dim[2] / 2
    z_max = dim[2] / 2
    xyz_range = [
        np.linspace(y_min, y_max, num_elements[1]),
        np.linspace(x_max, x_min, num_elements[0]),
        np.linspace(z_min, z_max, num_elements[2]),
    ]
    # xyz_range = [np.linspace(-dim[i] / 2, dim[i] / 2, num=num_elements[i]) for i in range(len(dim))]
    query_points = np.stack(np.meshgrid(*xyz_range), axis=-1).astype(np.float32)

    # Compute signed distance and occupancy
    sdf, _ = compute_signed_distance_and_closest_geometry(scene, query_points)

    # Reshape to a 3D grid
    sdf = sdf.reshape(*num_elements)

    return sdf


def visualize_sdf(sdf: np.ndarray):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    print("sdf ", sdf.shape)
    # We can visualize a slice of the grids directly with matplotlib
    fig, axes = plt.subplots(1, 1)
    # Create the initial image and color bar
    fake_array = np.zeros_like(sdf[:, :, 0])
    fake_array[0] = -1.0
    fake_array[-1] = 2.0
    im = axes.imshow(fake_array, origin="lower")
    # im = axes.imshow(sdf[:, :, 0], origin="lower")
    colorbar = fig.colorbar(im, ax=axes)

    # Define the animation function
    def update(frame):
        axes.clear()
        im = axes.imshow(sdf[:, :, frame], origin="lower")
        axes.set_title(f"Slice {frame+1} of {sdf.shape[2]}")

    # Create the animation
    anim = FuncAnimation(fig, update, frames=sdf.shape[2], interval=100, repeat=True)

    # Display the animation
    plt.show()
