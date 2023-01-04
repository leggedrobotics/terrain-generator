import trimesh
import numpy as np
from mesh_parts.mesh_parts_cfg import MeshPartsCfg, WallMeshPartsCfg, StairMeshPartsCfg
from mesh_parts.mesh_utils import merge_meshes, rotate_mesh, flip_mesh, ENGINE, get_height_array_of_mesh


def create_floor(cfg: MeshPartsCfg):
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0]
    floor = trimesh.creation.box(dims, pose)
    return floor


def create_standard_wall(cfg: WallMeshPartsCfg, edge: str = "bottom"):
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
            -cfg.dim[0] / 4.0, # + cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "bottom_right":
        dim = [cfg.dim[0] / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [
            cfg.dim[0] / 4.0, # - cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "right_bottom":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
        pos = [
            cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
            -cfg.dim[1] / 4.0, # + cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    elif edge == "right_up":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0, cfg.wall_height]
        pos = [
            cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0,
            cfg.dim[1] / 4.0, # - cfg.wall_thickness / 2.0,
            -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0,
        ]
    else:
        raise ValueError(f"Edge {edge} is not defined.")

    pose = np.eye(4)
    pose[:3, -1] = pos
    wall = trimesh.creation.box(dim, pose)
    return wall


def create_door(cfg: WallMeshPartsCfg, door_direction: str = "up"):
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


def create_wall_mesh(cfg: WallMeshPartsCfg):
    # Create the vertices of the wall
    floor = create_floor(cfg)
    mesh = floor
    for wall_edges in cfg.wall_edges:
        wall = create_standard_wall(cfg, wall_edges)
        # wall = get_wall_with_door(cfg, wall_edges)
        mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)
    if cfg.create_door:
        door = create_door(cfg, cfg.door_direction)
        mesh = trimesh.boolean.difference([mesh, door], engine=ENGINE)
    return mesh


def create_standard_stairs(cfg: StairMeshPartsCfg.Stair):
    """Create a standard stair with a given number of steps and a given height. This will fill bottom."""
    n_steps = cfg.n_steps
    step_height = cfg.total_height / n_steps
    step_depth = cfg.step_depth
    residual_depth = cfg.dim[1] - n_steps * step_depth
    mesh = trimesh.Trimesh()
    stair_start_pos = np.array([0.0, -cfg.dim[1] / 2.0, -cfg.dim[2] / 2.0])
    current_pos = stair_start_pos
    if cfg.add_residual_side_up is False:
        dims = np.array([cfg.step_width, residual_depth, cfg.floor_thickness + cfg.height_offset])
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        current_pos += np.array([0.0, dims[1], 0.0])
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    for n in range(n_steps):
        dims = [cfg.step_width, cfg.step_depth, (n + 1) * step_height + cfg.height_offset]
        pos = current_pos + np.array([0, dims[1] / 2.0, dims[2] / 2.0 - step_height])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        current_pos += np.array([0.0, dims[1], 0.0])
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    if cfg.add_residual_side_up is True:
        dims = np.array([cfg.step_width, residual_depth, n_steps * step_height + cfg.height_offset])
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    return mesh


# def create_standard_stairs_bk(cfg: StairMeshPartsCfg.Stair):
#     n_steps = int(cfg.total_height // cfg.step_height)
#     step_height = cfg.total_height / n_steps
#     step_depth = cfg.dim[1] / n_steps
#     mesh = trimesh.Trimesh()
#     # create stairs with up direction.
#     dim = np.array([cfg.step_width, cfg.step_depth * n_steps, step_height * n_steps])
#     if "up" in cfg.attach_side:
#         dz = cfg.dim[2] - cfg.floor_thickness - cfg.total_height
#         dim[2] += dz
#     # stair_start_pos = np.array([0.0, -n_steps * cfg.step_depth / 2.0, n_steps * step_height / 2.0])
#     stair_start_pos = np.array([0.0, -dim[1] / 2.0 + cfg.step_depth / 2.0, -dim[2] / 2.0])
#     for n in range(n_steps):
#         if cfg.fill_bottom:
#             dims = [cfg.step_width, cfg.step_depth, (n + 1) * step_height]
#             if "up" in cfg.attach_side:
#                 dz = cfg.dim[2] - cfg.floor_thickness - cfg.total_height
#                 dims[2] += dz
#             pos = [0, n * cfg.step_depth, dims[2] / 2.0]
#         else:
#             if n == 0:
#                 dims = [cfg.step_width, cfg.step_depth, step_height]
#                 pos = [0, n * cfg.step_depth, step_height / 2.0]
#             else:
#                 dims = [cfg.step_width, cfg.step_depth, step_height * 2.0]
#                 pos = [0, n * cfg.step_depth, (n + 1) * step_height - step_height]
#         pose = np.eye(4)
#         print("n, pos ", n, pos)
#         print("n, s + pos ", n, stair_start_pos + pos, stair_start_pos + pos - dim / 2, stair_start_pos + pos + dim / 2)
#         pose[:3, -1] = stair_start_pos + pos
#         step = trimesh.creation.box(dims, pose)
#         # print("step ", step)
#         # step.show()
#         if n == 0:
#             mesh = step
#         else:
#             mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
#         # mesh.show()
# 
#     # Fill in gaps
#     gap_dims = np.array([cfg.dim[0] - dim[0], cfg.dim[1] - dim[1], cfg.dim[2] - dim[2]])
#     # stair_start_pos = np.array([0.0, -dim[1] / 2.0 + cfg.step_depth / 2.0, -dim[2] / 2.0])
#     print("gap ", cfg.gap_direction)
#     if cfg.gap_direction == "down" and "up" in cfg.attach_side:
#         if cfg.fill_bottom:
#             gap_dim = [cfg.step_width, gap_dims[1], step_height]
#             dz = cfg.dim[2] - cfg.floor_thickness - cfg.total_height
#             gap_dim[2] += dz
#             gap_pos = [0, -dim[1] / 2.0 - gap_dim[1], gap_dim[2] / 2.0]
#         else:
#             gap_dims = [cfg.step_width, gap_dims[1], step_height]
#             gap_pos = [0, n_steps * cfg.step_depth - gap_dims[1] / 4.0, dim[2] - step_height]
#         pose = np.eye(4)
#         pose[:3, -1] = gap_pos + stair_start_pos
#         gap = trimesh.creation.box(gap_dim, pose)
#         mesh = merge_meshes([mesh, gap], cfg.minimal_triangles)
#     elif cfg.gap_direction == "up":
#         print("fill in gap")
#         if cfg.fill_bottom:
#             print("fill bottom")
#             gap_dims = [cfg.step_width, gap_dims[1], dim[2]]
#             gap_pos = [0, n_steps * cfg.step_depth - gap_dims[1] / 4.0, gap_dims[2] / 2.0]
#         else:
#             gap_dims = [cfg.step_width, gap_dims[1], step_height * 2.0]
#             gap_pos = [0, n_steps * cfg.step_depth - gap_dims[1] / 2.0, dim[2] - step_height]
#         pose = np.eye(4)
#         pose[:3, -1] = gap_pos + stair_start_pos
#         gap = trimesh.creation.box(gap_dims, pose)
#         mesh = merge_meshes([mesh, gap], cfg.minimal_triangles)
# 
#     print("dim ", dim)
#     if cfg.direction == "front":
#         mesh = mesh
#     elif cfg.direction == "left":
#         mesh = rotate_mesh(mesh, 90)
#         dim = dim[np.array([1, 0, 2])]
#     elif cfg.direction == "back":
#         mesh = rotate_mesh(mesh, 180)
#     elif cfg.direction == "right":
#         mesh = rotate_mesh(mesh, 270)
#         dim = dim[np.array([1, 0, 2])]
#     print("dim ", dim)
#     print("cfg dim ", cfg.dim)
#     if "left" in cfg.attach_side:
#         mesh.apply_translation([-cfg.dim[0] / 2.0 + dim[0] / 2.0, 0, 0])
#     if "right" in cfg.attach_side:
#         mesh.apply_translation([cfg.dim[0] / 2.0 - dim[0] / 2.0, 0, 0])
#     if "front" in cfg.attach_side:
#         mesh.apply_translation([0, cfg.dim[1] / 2.0 - dim[1] / 2.0, 0])
#     if "back" in cfg.attach_side:
#         mesh.apply_translation([0, -cfg.dim[1] / 2.0 + dim[1] / 2.0, 0])
#     if "up" in cfg.attach_side:
#         mesh.apply_translation([0, 0, cfg.dim[2] / 2.0 - dim[2] / 2.0])
#     if "bottom" in cfg.attach_side:
#         mesh.apply_translation([0, 0, -cfg.dim[2] / 2.0 + dim[2] / 2.0 + cfg.floor_thickness])
# 
#     # Fill the gaps
#     # if cfg.gap_direction == "left":
#     #     gap_box = trimesh.creation.box([cfg.dim[0], gaps[1], cfg.dim[2]], [0, -gaps[1] / 2.0, 0])
#     # if cfg.direction == "left" and "left" in cfg.attach_side:
# 
#     return mesh


def create_stairs(cfg: StairMeshPartsCfg.Stair):
    if cfg.stair_type == "standard":
        mesh = create_standard_stairs(cfg)

    dim = np.array([cfg.step_width, cfg.dim[1], cfg.total_height])
    if cfg.direction == "front":
        mesh = mesh
    elif cfg.direction == "left":
        mesh = rotate_mesh(mesh, 90)
        dim = dim[np.array([1, 0, 2])]
    elif cfg.direction == "back":
        mesh = rotate_mesh(mesh, 180)
    elif cfg.direction == "right":
        mesh = rotate_mesh(mesh, 270)
        dim = dim[np.array([1, 0, 2])]

    if "left" in cfg.attach_side:
        mesh.apply_translation([-cfg.dim[0] / 2.0 + dim[0] / 2.0, 0, 0])
    if "right" in cfg.attach_side:
        mesh.apply_translation([cfg.dim[0] / 2.0 - dim[0] / 2.0, 0, 0])
    if "front" in cfg.attach_side:
        mesh.apply_translation([0, cfg.dim[1] / 2.0 - dim[1] / 2.0, 0])
    if "back" in cfg.attach_side:
        mesh.apply_translation([0, -cfg.dim[1] / 2.0 + dim[1] / 2.0, 0])
    return mesh


def create_stairs_mesh(cfg: StairMeshPartsCfg):
    mesh = create_floor(cfg)
    for stair in cfg.stairs:
        stairs = create_stairs(stair)
        mesh = merge_meshes([mesh, stairs], cfg.minimal_triangles)
    if cfg.wall is not None:
        for wall_edges in cfg.wall.wall_edges:
            wall = create_standard_wall(cfg.wall, wall_edges)
            mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)

    return mesh


if __name__ == "__main__":
    # cfg = WallMeshPartsCfg(wall_edges=("left", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    # #
    # cfg = WallMeshPartsCfg(wall_edges=("up", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    #
    # cfg = WallMeshPartsCfg(wall_edges=("right", "bottom"))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()

    cfg = WallMeshPartsCfg(wall_edges=("bottom_right", "right_bottom"))
    mesh = create_wall_mesh(cfg)
    mesh.show()
    #
    # for i in range(10):
    #     cfg = WallMeshPartsCfg(wall_edges=("middle_right", "middle_left"), door_direction="up")
    #     mesh = create_wall_mesh(cfg)
    #     mesh.show()

    # cfg = StairMeshPartsCfg()
    # mesh = create_stairs(cfg)
    # mesh.show()
    #
    # cfg = StairMeshPartsCfg()
    # mesh = create_stairs(cfg.stairs[0])
    # mesh.show()

    stair_straight = StairMeshPartsCfg(
        name="stair_s",
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=2.0,
                # step_height=0.15,
                step_depth=0.3,
                total_height=1.0,
                stair_type="standard",
                direction="up",
                add_residual_side_up=True,
                attach_side="front",
                add_rail=False,
            ),
        ),
        # wall=WallMeshPartsCfg(
        #     name="wall",
        #     wall_edges=("middle_left", "middle_right"),
        #     )
    )
    # from mesh_parts.mesh_parts_cfg import StairPattern
    # pattern = StairPattern(name="stairs")
    mesh = create_stairs_mesh(stair_straight)
    mesh.show()
    print(get_height_array_of_mesh(mesh, stair_straight.dim, 5))

    # stair_straight = StairMeshPartsCfg(
    #     name="stair_s",
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    #     stairs=(
    #         StairMeshPartsCfg.Stair(
    #             step_width=1.0,
    #             # step_height=0.15,
    #             step_depth=0.3,
    #             total_height=1.0,
    #             stair_type="standard",
    #             direction="up",
    #             gap_direction="down",
    #             add_residual_side_up=False,
    #             attach_side="front_right",
    #             add_rail=False,
    #             fill_bottom=True,
    #         ),
    #     ),
    # )
    # # from mesh_parts.mesh_parts_cfg import StairPattern
    # # pattern = StairPattern(name="stairs")
    # mesh = create_stairs_mesh(stair_straight)
    # mesh.show()
    # 
    # stair_straight = StairMeshPartsCfg(
    #     name="stair_s",
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    #     stairs=(
    #         StairMeshPartsCfg.Stair(
    #             step_width=1.0,
    #             # step_height=0.15,
    #             step_depth=0.3,
    #             total_height=1.0,
    #             stair_type="standard",
    #             direction="up",
    #             gap_direction="up",
    #             add_residual_side_up=True,
    #             height_offset=1.0,
    #             attach_side="front_right",
    #             add_rail=False,
    #             fill_bottom=True,
    #         ),
    #     ),
    # )
    # # from mesh_parts.mesh_parts_cfg import StairPattern
    # # pattern = StairPattern(name="stairs")
    # mesh = create_stairs_mesh(stair_straight)
    # mesh.show()
