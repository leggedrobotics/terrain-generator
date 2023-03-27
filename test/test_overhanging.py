import numpy as np
import trimesh

from trimesh_tiles.mesh_parts.mesh_parts_cfg import WallMeshPartsCfg
from trimesh_tiles.mesh_parts.overhanging_parts import create_overhanging_boxes, generate_wall_from_array

from utils import get_height_array_of_mesh


def test_generate_wall_from_array(visualize):
    array = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.1, wall_height=2.0)
    mesh = generate_wall_from_array(cfg)

    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)
    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()

    array = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.4, wall_height=3.0)
    mesh = generate_wall_from_array(cfg)
    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)

    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()

    array = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.6, wall_height=1.0)
    mesh = generate_wall_from_array(cfg)
    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)

    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()

    array = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.2, wall_height=0.1)
    mesh = generate_wall_from_array(cfg)
    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)

    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()

    array = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.4, wall_height=3.0)
    mesh = generate_wall_from_array(cfg)

    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)

    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()
