import numpy as np
import trimesh

from utils import flip_mesh, rotate_mesh, get_height_array_of_mesh, merge_meshes, compute_sdf, visualize_sdf


def test_flip_mesh():
    mesh = trimesh.creation.box([1, 1, 0.1], trimesh.transformations.translation_matrix([0, 0, 0]))
    box = trimesh.creation.box([0.5, 0.5, 0.5], trimesh.transformations.translation_matrix([-0.25, -0.25, 0.25]))
    mesh = merge_meshes([mesh, box])
    # mesh.show()

    flipped_mesh = flip_mesh(mesh, "x")
    # flipped_mesh.show()
    assert np.allclose(flipped_mesh.vertices[:, 0], -mesh.vertices[:, 0])

    flipped_mesh = flip_mesh(mesh, "y")
    # flipped_mesh.show()
    assert np.allclose(flipped_mesh.vertices[:, 1], -mesh.vertices[:, 1])


def test_rotate_mesh():
    mesh = trimesh.creation.box([1, 1, 0.1], trimesh.transformations.translation_matrix([0, 0, 0]))
    box = trimesh.creation.box([0.5, 0.5, 0.5], trimesh.transformations.translation_matrix([-0.25, -0.25, 0.25]))
    mesh = merge_meshes([mesh, box])
    # mesh.show()

    rotated_mesh = rotate_mesh(mesh, 90)
    # rotated_mesh.show()
    # assert np.allclose(rotated_mesh.vertices[:, 0], mesh.vertices[:, 1])
    # assert np.allclose(rotated_mesh.vertices[:, 1], -mesh.vertices[:, 0])

    rotated_mesh = rotate_mesh(mesh, 180)
    # rotated_mesh.show()
    # assert np.allclose(rotated_mesh.vertices[:, 0], -mesh.vertices[:, 0])
    # assert np.allclose(rotated_mesh.vertices[:, 1], -mesh.vertices[:, 1])

    rotated_mesh = rotate_mesh(mesh, 270)
    # rotated_mesh.show()
    # assert np.allclose(rotated_mesh.vertices[:, 0], -mesh.vertices[:, 1])
    # assert np.allclose(rotated_mesh.vertices[:, 1], mesh.vertices[:, 0])


def test_get_height_array():
    mesh = trimesh.creation.box([1, 1, 0.1], trimesh.transformations.translation_matrix([0, 0, -0.5 + 0.05]))
    box = trimesh.creation.box([0.5, 0.5, 0.5], trimesh.transformations.translation_matrix([-0.25, -0.25, -0.25]))
    mesh = merge_meshes([mesh, box])
    # mesh.show()

    height_array = get_height_array_of_mesh(mesh, [1, 1, 1], 5)
    print(height_array)

    array = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.1, 0.1],
        ]
    )

    assert height_array.shape == (5, 5)
    assert np.allclose(height_array, array)

    flipped_mesh = flip_mesh(mesh, "x")
    height_array = get_height_array_of_mesh(flipped_mesh, [1, 1, 1], 5)
    print(height_array)

    array = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.5, 0.5, 0.5],
        ]
    )
    assert height_array.shape == (5, 5)
    assert np.allclose(height_array, array)

    flipped_mesh = flip_mesh(mesh, "y")
    height_array = get_height_array_of_mesh(flipped_mesh, [1, 1, 1], 5)
    print(height_array)

    array = np.array(
        [
            [0.5, 0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    assert height_array.shape == (5, 5)
    assert np.allclose(height_array, array)


def test_compute_sdf(visualize):
    box = trimesh.creation.box([0.5, 0.5, 0.5], trimesh.transformations.translation_matrix([0.0, 0.0, 0.0]))
    box2 = trimesh.creation.box([0.2, 0.2, 0.2], trimesh.transformations.translation_matrix([0.0, 0.0, 0.35]))
    box3 = trimesh.creation.box([0.2, 0.2, 0.2], trimesh.transformations.translation_matrix([0.0, 0.35, 0.0]))
    box4 = trimesh.creation.box([0.2, 0.2, 0.2], trimesh.transformations.translation_matrix([0.0, 0.00, 0.0]))
    box += box2 + box3 + box4
    sdf = compute_sdf(mesh=box, dim=[2, 2, 1], resolution=0.1)
    if visualize:
        box.show()
        visualize_sdf(sdf)


def test_visualize_sdf(sdf_path):
    sdf = np.load(sdf_path)
    visualize_sdf(sdf)
