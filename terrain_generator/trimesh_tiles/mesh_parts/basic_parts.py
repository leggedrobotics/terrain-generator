#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import trimesh
import numpy as np
from .mesh_parts_cfg import (
    MeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    WallPartsCfg,
    CapsuleMeshPartsCfg,
    BoxMeshPartsCfg,
)
from ...utils import (
    merge_meshes,
    merge_two_height_meshes,
    convert_heightfield_to_trimesh,
    merge_two_height_meshes,
    get_height_array_of_mesh,
    ENGINE,
)


def create_floor(cfg: MeshPartsCfg, **kwargs):
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0 + cfg.height_offset]
    floor = trimesh.creation.box(dims, pose)
    return floor


def create_standard_wall(cfg: WallPartsCfg, edge: str = "bottom", **kwargs):
    if edge == "bottom":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [
            0,
            -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "up":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [
            0,
            cfg.dim[1] / 2.0 - cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "left":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [
            -cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "right":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [
            cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "middle_bottom":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
        pos = [
            0,
            -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "middle_up":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
        pos = [
            0,
            cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "middle_left":
        dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [
            -cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "middle_right":
        dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [
            cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "bottom_left":
        dim = [cfg.dim[0] / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [
            -cfg.dim[0] / 4.0,  # + cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "bottom_right":
        dim = [cfg.dim[0] / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [
            cfg.dim[0] / 4.0,  # - cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "right_bottom":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
        pos = [
            cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 4.0,  # + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "right_up":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
        pos = [
            cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
            cfg.dim[1] / 4.0,  # - cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    else:
        raise ValueError(f"Edge {edge} is not defined.")

    pose = np.eye(4)
    pose[:3, -1] = pos
    wall = trimesh.creation.box(dim, pose)
    return wall


def create_door(cfg: WallPartsCfg, door_direction: str = "up", **kwargs):
    if door_direction == "bottom" or door_direction == "up":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction == "left" or door_direction == "right":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction == "middle_bottom":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [
            0,
            -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0,
            -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0,
        ]
    elif door_direction == "middle_up":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [
            0,
            cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0,
            -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
        ]
    elif door_direction == "middle_left":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [
            -cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
        ]
    elif door_direction == "middle_right":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [
            cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0,
            0,
            -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
        ]
    else:
        return trimesh.Trimesh()

    pose = np.eye(4)
    pose[:3, -1] = pos
    door = trimesh.creation.box(dim, pose)
    return door


def create_wall_mesh(cfg: WallPartsCfg, **kwargs):
    # Create the vertices of the wall
    floor = create_floor(cfg)
    meshes = [floor]
    # mesh = floor
    for wall_edges in cfg.wall_edges:
        wall = create_standard_wall(cfg, wall_edges)
        # wall = get_wall_with_door(cfg, wall_edges)
        meshes.append(wall)
        # mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    if cfg.create_door:
        door = create_door(cfg, cfg.door_direction)
        mesh = trimesh.boolean.difference([mesh, door], engine=ENGINE)
    return mesh


def create_platform_mesh(cfg: PlatformMeshPartsCfg, **kwargs):
    meshes = []
    min_h = 0.0
    if cfg.add_floor:
        meshes.append(create_floor(cfg))
        min_h = cfg.floor_thickness

    arrays = [cfg.array]
    z_dim_arrays = [cfg.z_dim_array]

    # Additional arrays
    if cfg.arrays is not None:
        arrays += cfg.arrays
    if cfg.z_dim_arrays is not None:
        z_dim_arrays += cfg.z_dim_arrays

    for array, z_dim_array in zip(arrays, z_dim_arrays):
        dim_xy = [cfg.dim[0] / array.shape[0], cfg.dim[1] / array.shape[1]]
        for y in range(array.shape[1]):
            for x in range(array.shape[0]):
                if array[y, x] > min_h:
                    h = array[y, x]
                    dim = [dim_xy[0], dim_xy[1], h]
                    if cfg.use_z_dim_array:
                        z = z_dim_array[y, x]
                        if z > 0.0 and z < h:
                            dim = np.array([dim_xy[0], dim_xy[1], z_dim_array[y, x]])
                    pos = np.array(
                        [
                            x * dim[0] - cfg.dim[0] / 2.0 + dim[0] / 2.0,
                            -y * dim[1] + cfg.dim[1] / 2.0 - dim[1] / 2.0,
                            h - dim[2] / 2.0 - cfg.dim[2] / 2.0,
                        ]
                    )
                    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
                    meshes.append(box_mesh)
    if cfg.wall is not None:
        wall_mesh = create_wall_mesh(cfg.wall)
        meshes.append(wall_mesh)
        # mesh = merge_meshes([mesh, wall_mesh], False)
        # mesh.fill_holes()
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    mesh.fill_holes()
    return mesh


def create_from_height_map(cfg: HeightMapMeshPartsCfg, **kwargs):
    mesh = trimesh.Trimesh()
    height_map = cfg.height_map

    if cfg.fill_borders:
        mesh = create_floor(cfg)

    height_map = height_map.copy() - cfg.dim[2] / 2.0
    height_map_mesh = convert_heightfield_to_trimesh(
        height_map,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    bottom_height_map = height_map * 0.0 + cfg.floor_thickness - cfg.dim[2] / 2.0
    bottom_mesh = convert_heightfield_to_trimesh(
        bottom_height_map,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    height_map_mesh = merge_two_height_meshes(height_map_mesh, bottom_mesh)

    mesh = merge_meshes([mesh, height_map_mesh], False)
    if cfg.simplify:
        mesh = mesh.simplify_quadratic_decimation(cfg.target_num_faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


# def create_capsule_mesh(cfg: CapsuleMeshPartsCfg, **kwargs):
#     # Create the vertices of the wall
#     if cfg.add_floor:
#         floor = create_floor(cfg)
#         meshes = [floor]
#     else:
#         meshes = []
#     for i in range(len(cfg.radii)):
#         capsule = trimesh.creation.capsule(
#             radius=cfg.radii[i],
#             height=cfg.heights[i],
#             # transform=cfg.transformations[i],
#         )
#         t = cfg.transformations[i].copy()
#         t[2, 3] -= cfg.dim[2] / 2.0
#         capsule.apply_transform(t)
#         meshes.append(capsule)
#     mesh = merge_meshes(meshes, cfg.minimal_triangles)
#     return mesh


def create_box_mesh(cfg: BoxMeshPartsCfg, **kwargs):
    print("create box mesh!!!!")
    if cfg.add_floor:
        print("create floor")
        floor = create_floor(cfg)
        meshes = [floor]
    else:
        print("not create floor")
        meshes = []
    for i in range(len(cfg.box_dims)):
        t = cfg.transformations[i].copy()
        t[2, 3] -= cfg.dim[2] / 2.0
        box = trimesh.creation.box(
            cfg.box_dims[i],
            t,
        )
        # box.apply_transform(t)
        meshes.append(box)
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    return mesh


def create_random_mesh(cfg: CapsuleMeshPartsCfg, **kwargs):
    # Create the vertices of the wall
    if cfg.add_floor:
        floor = create_floor(cfg)
        meshes = [floor]
    else:
        meshes = []
    for i in range(len(cfg.radii)):
        capsule = trimesh.creation.capsule(
            radius=cfg.radii[i],
            height=cfg.heights[i],
            # transform=cfg.transformations[i],
        )
        t = cfg.transformations[i].copy()
        t[2, 3] -= cfg.dim[2] / 2.0
        capsule.apply_transform(t)
        meshes.append(capsule)
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    return mesh


def create_capsule_mesh(cfg: CapsuleMeshPartsCfg, mesh: trimesh.Trimesh = None, **kwargs):
    # Create the vertices of the wall
    meshes = []
    positions = []
    for i in range(len(cfg.radii)):
        capsule = trimesh.creation.capsule(
            radius=cfg.radii[i],
            height=cfg.heights[i],
            # transform=cfg.transformations[i],
        )
        t = cfg.transformations[i].copy()
        t[2, 3] -= cfg.dim[2] / 2.0
        positions.append(t[0:3, 3])
        capsule.apply_transform(t)
        meshes.append(capsule)

    # Get heights of each position of capsule
    if mesh is not None:
        positions = np.array(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        origins = np.stack([x, y, np.ones_like(x) * cfg.dim[2] * 2], axis=-1)
        # print("origins ", origins)
        vectors = np.stack([np.zeros_like(x), np.zeros_like(y), -np.ones_like(x)], axis=-1)
        # print("vectors ", vectors)
        # # do the actual ray- mesh queries
        points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=True)
        # print("points ", points)
        # print("index_ray ", index_ray)
        # print("index_tri ", index_tri)
        translations = []
        for idx in index_ray:
            # positions[idx, 2] += points[idx, 2]
            translations.append(cfg.dim[2] / 2.0 + points[idx, 2])
        # print("translations ", translations)
        for i, m in enumerate(meshes):
            m.apply_translation([0, 0, translations[i]])
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    return mesh


if __name__ == "__main__":

    positions = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    transformations = [trimesh.transformations.random_rotation_matrix() for i in range(len(positions))]
    for i in range(len(positions)):
        transformations[i][:3, -1] = positions[i]
    capsule_cfg = CapsuleMeshPartsCfg(
        radii=(0.1, 0.2, 0.3), heights=(0.4, 0.5, 0.6), transformations=tuple(transformations)
    )
    capsule_mesh = create_capsule_mesh(capsule_cfg)
    capsule_mesh.show()
    print(get_height_array_of_mesh(capsule_mesh, capsule_cfg.dim, 5))

    cfg = HeightMapMeshPartsCfg(height_map=np.ones((3, 3)) * 1.4, target_num_faces=50)
    mesh = create_from_height_map(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    mesh.show()

    # print("showed 1st ex")

    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    # Z = np.sin(X * 2 * np.pi) * np.cos(Y * 2 * np.pi)
    Z = np.sin(X * 2 * np.pi)
    Z = (Z + 1.0) * 0.2 + 0.2
    cfg = HeightMapMeshPartsCfg(height_map=Z, target_num_faces=3000, simplify=True)
    mesh = create_from_height_map(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    print("mesh faces ", mesh.faces.shape)
    mesh.show()

    cfg = PlatformMeshPartsCfg(
        array=np.array([[1, 0], [0, 0]]),
        z_dim_array=np.array([[0.5, 0], [0, 0]]),
        use_z_dim_array=True,
    )
    mesh = create_platform_mesh(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    mesh.show()

    cfg = PlatformMeshPartsCfg(
        name="platform_T",
        array=np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
    )
    mesh = create_platform_mesh(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    mesh.show()

    cfg = PlatformMeshPartsCfg(array=np.array([[1, 0], [0, 0]]))
    mesh = create_platform_mesh(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    mesh.show()
    cfg = PlatformMeshPartsCfg(array=np.array([[2, 2], [0, 0]]))
    mesh = create_platform_mesh(cfg)
    print(get_height_array_of_mesh(mesh, cfg.dim, 5))
    mesh.show()
