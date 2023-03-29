import numpy as np
import trimesh

from trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile, get_mesh_gen
from trimesh_tiles.mesh_parts.mesh_parts_cfg import FloatingBoxesPartsCfg, WallMeshPartsCfg, PlatformMeshPartsCfg
from trimesh_tiles.mesh_parts.overhanging_parts import create_overhanging_boxes, generate_wall_from_array

from utils import get_height_array_of_mesh


def test_generate_wall_from_array(visualize):
    array = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.1, wall_height=2.0)
    mesh = generate_wall_from_array(cfg)

    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)
    assert np.allclose(height_array, array)
    if visualize:
        print("array ", array)
        print("height array ", height_array)
        # mesh.show()

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

    array = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    cfg = WallMeshPartsCfg(connection_array=array, wall_thickness=0.4, wall_height=3.0)
    mesh = generate_wall_from_array(cfg)

    height_array = (get_height_array_of_mesh(mesh, cfg.dim, 3) > 0).astype(np.float32)

    if visualize:
        print("array ", array)
        print("height array ", height_array)
        mesh.show()
    assert np.allclose(height_array, array)


def test_overhanging_boxes(visualize):
    n = 6
    height_diff = 0.5
    array = np.zeros((n, n))
    array[:] = np.linspace(0, height_diff, n)
    array = array.T
    array[1:-1, 1:-1] += np.random.normal(0, 0.1, size=[array.shape[0] - 2, array.shape[1] - 2])
    # array[5, :] = height_diff
    cfg = PlatformMeshPartsCfg(
        name="floor",
        array=array,
        flips=(),
        weight=0.1,
        minimal_triangles=False,
    )
    mesh_gen = get_mesh_gen(cfg)
    mesh = mesh_gen(cfg)
    cfg = FloatingBoxesPartsCfg(
        name="floating_boxes", mesh=mesh, gap_mean=0.8, gap_std=0.1, box_grid_n=4, box_height=0.6
    )
    box_cfg = create_overhanging_boxes(cfg)
    box_gen = get_mesh_gen(box_cfg)
    box_mesh = box_gen(box_cfg)
    mesh += box_mesh
    if visualize:
        mesh.show()
