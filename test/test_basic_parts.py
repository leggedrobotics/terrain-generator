import numpy as np
import trimesh
from trimesh_tiles.mesh_parts.basic_parts import create_capsule_mesh, create_floor
from trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    WallMeshPartsCfg,
    CapsuleMeshPartsCfg,
)
from utils import (
    merge_meshes,
    merge_two_height_meshes,
    convert_heightfield_to_trimesh,
    merge_two_height_meshes,
    get_height_array_of_mesh,
    ENGINE,
)


def test_capsule():
    positions = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]), np.array([-0.5, 0.0, 0.0])]
    transformations = [trimesh.transformations.random_rotation_matrix() for i in range(len(positions))]
    for i in range(len(positions)):
        transformations[i][:3, -1] = positions[i]
    capsule_cfg = CapsuleMeshPartsCfg(
        radii=(0.1, 0.2, 0.3), heights=(0.4, 0.3, 0.4), transformations=tuple(transformations)
    )
    capsule_mesh = create_capsule_mesh(capsule_cfg)
    floor = create_floor(capsule_cfg)
    mesh = merge_meshes([floor, capsule_mesh])
    mesh.show()
    print(get_height_array_of_mesh(capsule_mesh, capsule_cfg.dim, 5))


def test_rail():
    # poles
    x = np.linspace(-1.0, 1.0, 10)
    positions = [np.array([x[i], 0.0, -1.0 + 0.10]) for i in range(len(x))]
    heights = [0.5 for i in range(len(positions))]
    radii = [0.02 for i in range(len(positions))]
    transformations = [np.eye(4) for i in range(len(positions))]

    # rail
    positions.append(np.array([-1.0, 0.0, -0.5 + 0.1]))
    heights.append(2.0)
    radii.append(0.02)
    transformations.append(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
    print("transformations", transformations)
    # positions = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]), np.array([-0.5, 0.0, 0.0])]
    # transformations = [np.eye(4) for i in range(len(positions))]
    for i in range(len(positions)):
        transformations[i][:3, -1] = positions[i]
    capsule_cfg = CapsuleMeshPartsCfg(
        radii=tuple(radii), heights=tuple(heights), transformations=tuple(transformations), minimal_triangles=False
    )
    print(capsule_cfg)
    capsule_mesh = create_capsule_mesh(capsule_cfg)
    capsule_mesh.show()
    floor = create_floor(capsule_cfg)
    mesh = merge_meshes([floor, capsule_mesh], False)
    mesh.show()
    print(get_height_array_of_mesh(capsule_mesh, capsule_cfg.dim, 5))
