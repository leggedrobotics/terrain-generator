import trimesh
import numpy as np
from mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
)
from mesh_parts.mesh_utils import (
    merge_meshes,
    merge_two_height_meshes,
    convert_heightfield_to_trimesh,
    merge_two_height_meshes,
    get_height_array_of_mesh,
)


def create_floor(cfg: MeshPartsCfg):
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0 + cfg.height_offset]
    floor = trimesh.creation.box(dims, pose)
    return floor


def create_platform_mesh(cfg: PlatformMeshPartsCfg):
    meshes = []
    min_h = 0.0
    if cfg.add_floor:
        meshes.append(create_floor(cfg))
        min_h = cfg.floor_thickness
    dim_xy = [cfg.dim[0] / cfg.array.shape[0], cfg.dim[1] / cfg.array.shape[1]]
    for y in range(cfg.array.shape[1]):
        for x in range(cfg.array.shape[0]):
            if cfg.array[y, x] > min_h:
                h = cfg.array[y, x]
                dim = [dim_xy[0], dim_xy[1], h]
                if cfg.use_z_dim_array:
                    z = cfg.z_dim_array[y, x]
                    if z > 0.0 and z < h:
                        dim = np.array([dim_xy[0], dim_xy[1], cfg.z_dim_array[y, x]])
                pos = np.array(
                    [
                        x * dim[0] - cfg.dim[0] / 2.0 + dim[0] / 2.0,
                        -y * dim[1] + cfg.dim[1] / 2.0 - dim[1] / 2.0,
                        h - dim[2] / 2.0 - cfg.dim[2] / 2.0,
                    ]
                )
                box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
                meshes.append(box_mesh)
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    return mesh


def create_from_height_map(cfg: HeightMapMeshPartsCfg):
    mesh = trimesh.Trimesh()
    height_map = cfg.height_map

    if cfg.fill_borders:
        mesh = create_floor(cfg)

    height_map -= cfg.dim[2] / 2.0
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


if __name__ == "__main__":
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
