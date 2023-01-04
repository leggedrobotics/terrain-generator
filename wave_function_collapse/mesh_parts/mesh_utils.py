import os
import numpy as np
import trimesh
from typing import Callable, Any
from dataclasses import asdict, is_dataclass
import open3d as o3d


ENGINE = "blender"
CACHE_DIR = "mesh_cache"
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
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    vectors = np.stack([np.zeros_like(xv), np.zeros_like(yv), -np.ones_like(xv)], axis=-1)
    # # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    array = np.zeros((num_points * num_points))
    array[index_ray] = points[:, 2]
    array = np.round(array, 1) + dim[2] / 2.0
    array = array.reshape(num_points, num_points)
    return array


def cfg_to_hash(cfg):
    """MD5 hash of a config."""
    import hashlib
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if isinstance(cfg, dict):
        encoded = json.dumps(cfg, sort_keys=True, cls=NpEncoder).encode()
    elif is_dataclass(cfg):
        encoded = json.dumps(asdict(cfg), sort_keys=True, cls=NpEncoder).encode()
    else:
        raise ValueError("cfg must be a dict or dataclass.")
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    # encoded = json.dumps(cfg, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_cached_mesh_gen(
    mesh_gen_fn: Callable[[Any], trimesh.Trimesh], cfg, verbose=False
) -> Callable[[], trimesh.Trimesh]:
    """Generate a mesh if there's no cache. If there's cache, load from cache."""
    code = cfg_to_hash(cfg)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if hasattr(cfg, "name"):
        name = cfg.name
    else:
        name = ""

    def mesh_gen() -> trimesh.Trimesh:
        if os.path.exists(os.path.join(CACHE_DIR, code + ".stl")):
            if verbose:
                print(f"Loading mesh {name} from cache {code}.stl ...")
            mesh = trimesh.load_mesh(os.path.join(CACHE_DIR, code + ".stl"))
        else:
            if verbose:
                print(f"Not loading {name} from cache, creating {code}.stl ...")
            mesh = mesh_gen_fn(cfg)
            mesh.export(os.path.join(CACHE_DIR, code + ".stl"))
        return mesh

    return mesh_gen


def visualize_mesh(mesh, save_path=None):
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
    print("R ", R)
    o3d_mesh.rotate(R, center=[0, 0, 0])
    o3d.visualization.draw_geometries([o3d_mesh])
    # vis.capture_screen_image(f"image_{i}.jpg")
