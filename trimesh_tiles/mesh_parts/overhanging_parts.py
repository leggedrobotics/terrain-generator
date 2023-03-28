import trimesh
import numpy as np
from utils import (
    merge_meshes,
    rotate_mesh,
    ENGINE,
    get_height_array_of_mesh,
    get_heights_from_mesh,
)
from .mesh_parts_cfg import (
    FloatingBoxesPartsCfg,
    WallMeshPartsCfg,
    OverhangingMeshPartsCfg,
    FloatingBoxesPartsCfg,
    PlatformMeshPartsCfg,
    # StairMeshPartsCfg,
)


def create_wall(width, height, depth):
    mesh = trimesh.creation.box([width, height, depth])
    return mesh


# TODO: finish this.
def create_horizontal_bar(width, height, depth):
    # mesh = trimesh.creation.box([width, height, depth])
    mesh = trimesh.creation.cylinder()
    return mesh


def generate_wall_from_array(cfg: WallMeshPartsCfg) -> trimesh.Trimesh:
    """generate wall mesh from connection array.
    Args: connection_array (np.ndarray) shape=(3, 3)
    Return: trimesh.Trimesh
    """
    assert cfg.connection_array.shape[0] == 3 and cfg.connection_array.shape[1] == 3
    grid_size = cfg.dim[0] / cfg.connection_array.shape[0]
    meshes = []
    wall_fn = create_wall
    for y in range(cfg.connection_array.shape[1]):
        for x in range(cfg.connection_array.shape[0]):
            if cfg.connection_array[x, y] > 0:
                pos = np.array([x * grid_size, y * grid_size, 0])
                pos[:2] += grid_size / 2.0 - cfg.dim[0] / 2.0
                if np.abs(pos[0]) > 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                    mesh = wall_fn(grid_size, cfg.wall_thickness, cfg.wall_height)
                    mesh.apply_translation(pos)
                    meshes.append(mesh)
                elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) > 1.0e-4:
                    mesh = wall_fn(cfg.wall_thickness, grid_size, cfg.wall_height)
                    mesh.apply_translation(pos)
                    meshes.append(mesh)
                elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                    # Get the coordinates of the neighboring walls
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if (
                            0 <= x + dx < cfg.connection_array.shape[0]
                            and 0 <= y + dy < cfg.connection_array.shape[1]
                            and cfg.connection_array[x + dx, y + dy] > 0
                        ):
                            neighbors.append((dx, dy))
                    if not neighbors:
                        continue

                    # Calculate the dimensions of the center wall
                    # width = grid_size + cfg.wall_thickness
                    # height = grid_size + cfg.wall_thickness
                    for dx, dy in neighbors:
                        width = grid_size / 2.0
                        height = grid_size / 2.0
                        depth = cfg.wall_height
                        p = pos.copy()
                        if dx == 1:
                            width += cfg.wall_thickness / 2.0
                            height = cfg.wall_thickness
                            p[0] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dx == -1:
                            width += cfg.wall_thickness / 2.0
                            height = cfg.wall_thickness
                            p[0] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dy == 1:
                            height += cfg.wall_thickness / 2.0
                            width = cfg.wall_thickness
                            p[1] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dy == -1:
                            height += cfg.wall_thickness / 2.0
                            width = cfg.wall_thickness
                            p[1] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        else:
                            continue

                        mesh = wall_fn(width, height, depth)
                        mesh.apply_translation(p)
                        meshes.append(mesh)
                else:
                    mesh = wall_fn(grid_size, grid_size, cfg.wall_height)
                    meshes.append(mesh)
    mesh = merge_meshes(meshes, minimal_triangles=cfg.minimal_triangles, engine=ENGINE)
    mesh = rotate_mesh(mesh, 270)  # This was required to match the connection array
    return mesh


def create_overhanging_boxes(cfg: FloatingBoxesPartsCfg, **kwargs):
    if cfg.mesh is not None:
        height_array = get_height_array_of_mesh(cfg.mesh, cfg.dim, cfg.box_grid_n)
    elif cfg.height_array is not None:
        height_array = cfg.height_array
    else:
        height_array = np.zeros((cfg.box_grid_n, cfg.box_grid_n))
    array = np.random.normal(cfg.gap_mean, cfg.gap_std, size=height_array.shape)
    array += height_array
    z_dim_array = np.ones_like(array) * cfg.box_height
    floating_array = array + z_dim_array

    overhanging_cfg = PlatformMeshPartsCfg(
        name="floating_boxes",
        dim=cfg.dim,
        array=floating_array,
        z_dim_array=z_dim_array,
        rotations=(90, 180, 270),
        flips=("x", "y"),
        weight=0.1,
        minimal_triangles=False,
        add_floor=False,
        use_z_dim_array=True,
    )
    return overhanging_cfg


# def create_standard_wall(cfg: WallMeshPartsCfg, edge: str = "bottom", **kwargs):
#     if edge == "bottom":
#         dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             0,
#             -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "up":
#         dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             0,
#             cfg.dim[1] / 2.0 - cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "left":
#         dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
#         pos = [
#             -cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "right":
#         dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
#         pos = [
#             cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "middle_bottom":
#         dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
#         pos = [
#             0,
#             -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "middle_up":
#         dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
#         pos = [
#             0,
#             cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "middle_left":
#         dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             -cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "middle_right":
#         dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "bottom_left":
#         dim = [cfg.dim[0] / 2.0, cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             -cfg.dim[0] / 4.0,  # + cfg.wall_thickness / 2.0,
#             -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "bottom_right":
#         dim = [cfg.dim[0] / 2.0, cfg.wall_thickness, cfg.wall_height]
#         pos = [
#             cfg.dim[0] / 4.0,  # - cfg.wall_thickness / 2.0,
#             -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "right_bottom":
#         dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
#         pos = [
#             cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
#             -cfg.dim[1] / 4.0,  # + cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     elif edge == "right_up":
#         dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
#         pos = [
#             cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
#             cfg.dim[1] / 4.0,  # - cfg.wall_thickness / 2.0,
#             -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
#         ]
#     else:
#         raise ValueError(f"Edge {edge} is not defined.")
#
#     pose = np.eye(4)
#     pose[:3, -1] = pos
#     wall = trimesh.creation.box(dim, pose)
#     return wall
#
#
# def create_door(cfg: WallPartsCfg, door_direction: str = "up", **kwargs):
#     if door_direction == "bottom" or door_direction == "up":
#         dim = [cfg.door_width, 2.0, cfg.door_height]
#         pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
#     elif door_direction == "left" or door_direction == "right":
#         dim = [2.0, cfg.door_width, cfg.door_height]
#         pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
#     elif door_direction == "middle_bottom":
#         dim = [2.0, cfg.door_width, cfg.door_height]
#         pos = [
#             0,
#             -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0,
#             -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0,
#         ]
#     elif door_direction == "middle_up":
#         dim = [2.0, cfg.door_width, cfg.door_height]
#         pos = [
#             0,
#             cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0,
#             -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
#         ]
#     elif door_direction == "middle_left":
#         dim = [cfg.door_width, 2.0, cfg.door_height]
#         pos = [
#             -cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
#         ]
#     elif door_direction == "middle_right":
#         dim = [cfg.door_width, 2.0, cfg.door_height]
#         pos = [
#             cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0,
#             0,
#             -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0,
#         ]
#     else:
#         return trimesh.Trimesh()
#
#     pose = np.eye(4)
#     pose[:3, -1] = pos
#     door = trimesh.creation.box(dim, pose)
#     return door
#
#
# def create_wall_mesh(cfg: WallPartsCfg, **kwargs):
#     # Create the vertices of the wall
#     floor = create_floor(cfg)
#     meshes = [floor]
#     # mesh = floor
#     for wall_edges in cfg.wall_edges:
#         wall = create_standard_wall(cfg, wall_edges)
#         # wall = get_wall_with_door(cfg, wall_edges)
#         meshes.append(wall)
#         # mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)
#     mesh = merge_meshes(meshes, cfg.minimal_triangles)
#     if cfg.create_door:
#         door = create_door(cfg, cfg.door_direction)
#         mesh = trimesh.boolean.difference([mesh, door], engine=ENGINE)
#     return mesh
