import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    MeshPartsCfg,
    WallMeshPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
)


@dataclass
class FloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        WallMeshPartsCfg(name="floor", dim=dim, wall_edges=(), weight=10.0),
        WallMeshPartsCfg(
            name="wall_s",
            dim=dim,
            wall_edges=("middle_left", "middle_right"),
            rotations=(90, 180, 270),
            flips=(),
            weight=2.0,
        ),
        WallMeshPartsCfg(
            name="wall_t",
            dim=dim,
            wall_edges=("middle_left", "middle_bottom"),
            rotations=(90, 180, 270),
            flips=(),
            weight=1.0,
        ),
        WallMeshPartsCfg(
            name="door_s",
            dim=dim,
            wall_edges=("middle_left", "middle_right"),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.2,
            door_direction="up",
            create_door=True,
        ),
        WallMeshPartsCfg(
            name="wall_s_e",
            dim=dim,
            wall_edges=("left",),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        # platform0: MeshPartsCfg = PlatformMeshPartsCfg(
        #     name="platform_1000",
        #     dim=dim,
        #     array=np.array([[1, 0], [0, 0]]),
        #     rotations=(90, 180, 270),
        #     flips=(),
        #     weight=0.1,
        # )
        PlatformMeshPartsCfg(
            name="platform_1100",
            dim=dim,
            array=np.array([[1, 1], [0, 0]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="platform_1110",
            dim=dim,
            array=np.array([[1, 1], [1, 0]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.10,
        ),
        PlatformMeshPartsCfg(
            name="platform_1111",
            dim=dim,
            array=np.array([[1, 1], [1, 1]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        # platform4: MeshPartsCfg = PlatformMeshPartsCfg(
        #     name="platform_2000",
        #     dim=dim,
        #     array=np.array([[2, 0], [0, 0]]),
        #     rotations=(90, 180, 270),
        #     flips=(),
        #     weight=0.1,
        # )
        PlatformMeshPartsCfg(
            name="platform_2200",
            dim=dim,
            array=np.array([[2, 2], [0, 0]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="platform_2220",
            dim=dim,
            array=np.array([[2, 2], [2, 0]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.10,
        ),
        PlatformMeshPartsCfg(
            name="platform_2222",
            dim=dim,
            array=np.array([[2, 2], [2, 2]]),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        StairMeshPartsCfg(
            name="stair_w",
            rotations=(90, 180, 270),
            flips=(),
            weight=1.0,
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
        ),
        StairMeshPartsCfg(
            name="stair_s",
            dim=dim,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=1.0,
            stairs=(
                StairMeshPartsCfg.Stair(
                    step_width=1.0,
                    step_depth=0.3,
                    total_height=1.0,
                    direction="up",
                    add_residual_side_up=False,
                    attach_side="front_right",
                    add_rail=False,
                ),
            ),
        ),
        StairMeshPartsCfg(
            name="stair_s_u",
            dim=dim,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=1.0,
            stairs=(
                StairMeshPartsCfg.Stair(
                    step_width=1.0,
                    step_depth=0.3,
                    total_height=1.0,
                    height_offset=1.0,
                    direction="up",
                    add_residual_side_up=True,
                    attach_side="front_right",
                    add_rail=False,
                ),
            ),
        ),
    )
    # wall_turn_edge: MeshPartsCfg = WallMeshPartsCfg(
    #     name="wall_t_e",
    #     dim=dim,
    #     wall_edges=("bottom_right",),
    #     rotations=(90, 180, 270),
    #     flips=("x", "y"),
    #     weight=0.1,
    #     door_direction="",
    # )
    # wall_turn_T: MeshPartsCfg = WallMeshPartsCfg(
    #     name="wall_T_e",
    #     dim=dim,
    #     wall_edges=("bottom_right", "right_bottom"),
    #     rotations=(90, 180, 270),
    #     flips=("x", "y"),
    #     weight=0.1,
    #     door_direction="",
    # )
    # wall_straight: MeshPartsCfg = WallMeshPartsCfg(name="wall_s", dim=dim, wall_edges=("up",), rotations=(90, 180, 270), flips=(), weight=2.0)
    # wall_turn: MeshPartsCfg = WallMeshPartsCfg(name="wall_t", dim=dim, wall_edges=("up", "right"), rotations=(90, 180, 270), flips=(), weight=1.0)
    # wall_straight_door: MeshPartsCfg = WallMeshPartsCfg(name="door_s", dim=dim, wall_edges=("up",), rotations=(90, 180, 270), flips=(), weight=0.2, door_direction="up", create_door=True)


# @dataclass
# class StairsPattern(MeshPattern):
#     dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
# floor_middle: MeshPartsCfg = WallMeshPartsCfg(
#     name="floor_middle", dim=dim, wall_edges=(), weight=0.1, height_offset=1.0
# )
# floor_top: MeshPartsCfg = WallMeshPartsCfg(name="floor_top", dim=dim, wall_edges=(), weight=0.1, height_offset=2.0)
#
# stair_straight_wall: MeshPartsCfg = StairMeshPartsCfg(
#         name="stair_s_w",
#         dim=dim,
#         rotations=(90, 180, 270),
#         flips=("x", "y"),
#         weight=0.1,
#         stairs=(
#             StairMeshPartsCfg.Stair(
#                 step_width=1.0,
#                 step_depth=0.3,
#                 total_height=1.0,
#                 direction="up",
#                 add_residual_side_up=True,
#                 attach_side="front_right",
#                 add_rail=False,
#                 ),
#             ),
#         wall=WallMeshPartsCfg(
#             name="wall",
#             wall_edges=("right",),
#             )
#         )
# stair_straight_up_wall: MeshPartsCfg = StairMeshPartsCfg(
#         name="stair_s_u_w",
#         dim=dim,
#         rotations=(90, 180, 270),
#         flips=("x", "y"),
#         weight=0.1,
#         stairs=(
#             StairMeshPartsCfg.Stair(
#                 step_width=1.0,
#                 step_depth=0.3,
#                 total_height=1.0,
#                 height_offset=1.0,
#                 direction="up",
#                 add_residual_side_up=True,
#                 attach_side="front_right",
#                 add_rail=False,
#                 ),
#             ),
#         wall=WallMeshPartsCfg(
#             name="wall",
#             wall_edges=("right",),
#             )
#         )
# stair_turn: MeshPartsCfg = StairMeshPartsCfg(
#         name="stair_t",
#         dim=dim,
#         stair_start_direction="bottom_right",
#         stair_end_direction="up_left",
#         rotations=(90, 180, 270),
#         flips=("x",),
#         weight=0.1,
#         stair=StairMeshPartsCfg.Stair(
#             step_width=1.0,
#             step_height=0.2,
#             step_depth=0.4,
#             total_height=1.0,
#             stair_type="standard",
#             stair_start_direction="bottom_right",
#             stair_end_direction="up_right",
#             add_rail=False,
#             fill_bottom=False),
#         )
# wall_straight: MeshPartsCfg = WallMeshPartsCfg(name="wall_s", dim=dim, wall_edges=("middle_left", "middle_right"), rotations=(90, 180, 270), flips=(), weight=2.0, door_direction="up")
# wall_turn: MeshPartsCfg = WallMeshPartsCfg(name="wall_t", dim=dim, wall_edges=("middle_left", "middle_bottom"), rotations=(90, 180, 270), flips=(), weight=1.0, door_direction="")
# wall_straight_door: MeshPartsCfg = WallMeshPartsCfg(name="door_s", dim=dim, wall_edges=("middle_left", "middle_right"), rotations=(90, 180, 270), flips=(), weight=0.2, door_direction="up", create_door=True)
