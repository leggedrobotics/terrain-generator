import numpy as np
import trimesh


ENGINE = "blender"
# ENGINE = "scad"


def merge_meshes(meshes, minimal_triangles=False):
    if minimal_triangles:
        meshes = trimesh.boolean.union(meshes, engine=ENGINE)
    else:
        meshes = trimesh.util.concatenate(meshes)
    return meshes


def flip_mesh(mesh, direction):
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


def rotate_mesh(mesh, deg):
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


def get_height_array_of_mesh(mesh, dim, num_points):
    # intersects_location requires origins to be the same shape as vectors
    x = np.linspace(-dim[0] / 2.0, dim[0] / 2.0, num_points)
    y = np.linspace(dim[1] / 2.0, -dim[1] / 2.0, num_points)
    # y = np.linspace(dim[1] / 2.0, -dim[1] / 2.0, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    vectors = np.stack([np.zeros_like(xv), np.zeros_like(yv), -np.ones_like(xv)], axis=-1)
    # # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)
    array = np.zeros((num_points * num_points))
    array[index_ray] = points[:, 2]
    array = np.round(array, 1) + dim[2] / 2.0
    array = array.reshape(num_points, num_points)
    return array
