import trimesh
import numpy as np
from utils import (
    merge_meshes,
    yaw_rotate_mesh,
    ENGINE,
    get_height_array_of_mesh,
)
from .mesh_parts_cfg import (
    WallPartsCfg,
    StairMeshPartsCfg,
)
from .basic_parts import create_floor


def create_standard_wall(cfg: WallPartsCfg, edge: str = "bottom"):
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


def create_door(cfg: WallPartsCfg, door_direction: str = "up"):
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


def create_wall_mesh(cfg: WallPartsCfg):
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
    residual_depth = cfg.dim[1] - (n_steps + 1) * step_depth
    mesh = trimesh.Trimesh()
    stair_start_pos = np.array([0.0, -cfg.dim[1] / 2.0, -cfg.dim[2] / 2.0])
    current_pos = stair_start_pos
    if cfg.add_residual_side_up is False:
        dims = np.array([cfg.step_width, residual_depth, cfg.height_offset])
        current_pos += np.array([0.0, dims[1], 0.0])
        if cfg.height_offset != 0.0:
            pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
            step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
            mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    for n in range(n_steps + 1):
        if n == 0:
            if cfg.height_offset > 0:
                dims = [cfg.step_width, cfg.step_depth, cfg.height_offset]
            else:
                dims = [cfg.step_width, cfg.step_depth, cfg.floor_thickness]
        else:
            dims = [cfg.step_width, cfg.step_depth, n * step_height + cfg.height_offset]
        pos = current_pos + np.array([0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        current_pos += np.array([0.0, dims[1], 0.0])
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    if cfg.add_residual_side_up is True:
        dims = np.array([cfg.step_width, residual_depth, n_steps * step_height + cfg.height_offset])
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    return mesh


def create_stairs(cfg: StairMeshPartsCfg.Stair):
    if cfg.stair_type == "standard":
        mesh = create_standard_stairs(cfg)

    dim = np.array([cfg.step_width, cfg.dim[1], cfg.total_height])
    if cfg.direction == "front":
        mesh = mesh
    elif cfg.direction == "left":
        mesh = yaw_rotate_mesh(mesh, 90)
        dim = dim[np.array([1, 0, 2])]
    elif cfg.direction == "back":
        mesh = yaw_rotate_mesh(mesh, 180)
    elif cfg.direction == "right":
        mesh = yaw_rotate_mesh(mesh, 270)
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

    # cfg = WallMeshPartsCfg(wall_edges=("bottom_right", "right_bottom"))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
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

    # stair_straight = StairMeshPartsCfg(
    #     name="stair_s",
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    #     stairs=(
    #         StairMeshPartsCfg.Stair(
    #             step_width=2.0,
    #             # step_height=0.15,
    #             step_depth=0.3,
    #             total_height=1.0,
    #             stair_type="standard",
    #             direction="up",
    #             add_residual_side_up=True,
    #             attach_side="front",
    #             add_rail=False,
    #         ),
    #     ),
    #     # wall=WallMeshPartsCfg(
    #     #     name="wall",
    #     #     wall_edges=("middle_left", "middle_right"),
    #     #     )
    # )
    stair_wide = StairMeshPartsCfg(
        name="stair_w",
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=2.0,
                step_depth=0.3,
                total_height=1.0,
                stair_type="standard",
                direction="up",
                add_residual_side_up=False,
                attach_side="front",
                add_rail=False,
            ),
        ),
    )
    # from mesh_parts.mesh_parts_cfg import StairPattern
    # pattern = StairPattern(name="stairs")
    mesh = create_stairs_mesh(stair_wide)
    mesh.show()
    print(get_height_array_of_mesh(mesh, stair_wide.dim, 5))

    stair_straight = StairMeshPartsCfg(
        name="stair_s",
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=1.0,
                # step_height=0.15,
                step_depth=0.3,
                total_height=1.0,
                height_offset=1.0,
                stair_type="standard",
                direction="up",
                add_residual_side_up=True,
                attach_side="front_right",
                add_rail=False,
            ),
        ),
    )
    mesh = create_stairs_mesh(stair_straight)
    mesh.show()
    print(get_height_array_of_mesh(mesh, stair_straight.dim, 5))
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
