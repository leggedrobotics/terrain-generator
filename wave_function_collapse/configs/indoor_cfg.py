import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from numpy.random import f

from mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    MeshPartsCfg,
    WallMeshPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
)
from mesh_parts.rough_parts import generate_perlin_tile_configs

from patterns.pattern_generator import (
    generate_walls,
    generate_floating_boxes,
    generate_narrow,
    generate_platforms,
    generate_ramp_parts,
    generate_stair_parts,
    generate_stepping_stones,
)


@dataclass
class IndoorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        tuple(generate_walls(dim))
        + tuple(generate_platforms(name="platform_1", dim=dim, max_h=1.0, min_h=0.0, weight=0.5))
        + tuple(generate_platforms(name="platform_2", dim=dim, max_h=2.0, min_h=0.0, weight=0.5))
        + tuple(generate_platforms(name="platform_2_1", dim=dim, max_h=2.0, min_h=1.0, weight=0.5))
        + tuple(generate_platforms(name="platform_0.5", dim=dim, max_h=0.5, min_h=0.0, weight=0.5))
        + tuple(generate_platforms(name="platform_1_0.5", dim=dim, max_h=1.0, min_h=0.5, weight=0.5))
        + tuple(generate_stepping_stones(name="stepping_1", dim=dim, max_h=1.0, min_h=0.0, weight=1.2))
        + tuple(generate_stepping_stones(name="stepping_2", dim=dim, max_h=2.0, min_h=0.0, weight=1.2))
        + tuple(generate_stepping_stones(name="stepping_2_1", dim=dim, max_h=2.0, min_h=1.0, weight=1.2))
        + tuple(generate_stepping_stones(name="stepping_0.5", dim=dim, max_h=0.5, min_h=0.0, weight=1.2))
        + tuple(generate_stepping_stones(name="stepping_1_0.5", dim=dim, max_h=1.0, min_h=0.5, weight=1.2))
        + tuple(generate_narrow(name="narrow_1", dim=dim, max_h=1.0, min_h=0.0, weight=0.2))
        + tuple(generate_narrow(name="narrow_2", dim=dim, max_h=2.0, min_h=0.0, weight=0.2))
        + tuple(generate_narrow(name="narrow_2_1", dim=dim, max_h=2.0, min_h=1.0, weight=0.2))
        + tuple(generate_narrow(name="narrow_0.5", dim=dim, max_h=0.5, min_h=0.0, weight=0.2))
        + tuple(generate_narrow(name="narrow_1_0.5", dim=dim, max_h=1.0, min_h=0.5, weight=0.2))
        + tuple(generate_floating_boxes(n=20, dim=dim, seed=seed, array_shape=[5, 5], weight=0.05))
        + tuple(generate_stair_parts(name="stair", dim=dim, seed=seed, array_shape=[15, 15], weight=1.0, depth_num=2))
        + tuple(
            generate_stair_parts(
                name="stair_offset", dim=dim, seed=seed, array_shape=[15, 15], weight=2.0, depth_num=2, offset=1.0
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low", dim=dim, total_height=0.5, seed=seed, array_shape=[15, 15], weight=1.0, depth_num=2
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset",
                dim=dim,
                total_height=0.5,
                offset=0.5,
                seed=seed,
                array_shape=[15, 15],
                weight=1.0,
                depth_num=2,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset_1",
                dim=dim,
                total_height=0.5,
                offset=1.0,
                seed=seed,
                array_shape=[15, 15],
                weight=1.0,
                depth_num=2,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset_2",
                dim=dim,
                total_height=0.5,
                offset=1.5,
                seed=seed,
                array_shape=[15, 15],
                weight=1.0,
                depth_num=2,
            )
        )
        + tuple(
            generate_ramp_parts(
                name="ramp",
                dim=dim,
                seed=seed,
                array_shape=[30, 30],
                total_height=1.0,
                offset=0.00,
                weight=1.0,
                depth_num=1,
            )
        )
        + tuple(
            generate_ramp_parts(
                name="ramp_low",
                dim=dim,
                seed=seed,
                array_shape=[30, 30],
                total_height=0.5,
                offset=0.00,
                weight=1.0,
                depth_num=1,
            )
        )
        # + tuple(generate_perlin_tile_configs(name="perlin_0", dim=dim, seed=seed, weight=1.2))
        # + tuple(generate_perlin_tile_configs(name="perlin_0.5", dim=dim, seed=seed, weight=1.2, offset=0.5))
        # + tuple(generate_perlin_tile_configs(name="perlin_1", dim=dim, seed=seed, weight=1.2, offset=1.0))
    )
    # platform0: MeshPartsCfg = PlatformMeshPartsCfg(
    # PlatformMeshPartsCfg(
    #     name="platform_2.0",
    #     dim=dim,
    #     array=np.ones((5, 5)) * 2.0,
    #     ),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    #     StairMeshPartsCfg(
    #         name="stair_w",
    #         rotations=(90, 180, 270),
    #         flips=(),
    #         weight=1.0,
    #         stairs=(
    #             StairMeshPartsCfg.Stair(
    #                 step_width=2.0,
    #                 step_depth=0.3,
    #                 total_height=1.0,
    #                 stair_type="standard",
    #                 direction="up",
    #                 add_residual_side_up=False,
    #                 attach_side="front",
    #                 add_rail=False,
    #             ),
    #         ),
    #     ),
    #     StairMeshPartsCfg(
    #         name="stair_s",
    #         dim=dim,
    #         rotations=(90, 180, 270),
    #         flips=("x", "y"),
    #         weight=1.0,
    #         stairs=(
    #             StairMeshPartsCfg.Stair(
    #                 step_width=1.0,
    #                 step_depth=0.3,
    #                 total_height=1.0,
    #                 direction="up",
    #                 add_residual_side_up=False,
    #                 attach_side="front_right",
    #                 add_rail=False,
    #             ),
    #         ),
    #     ),
    #     StairMeshPartsCfg(
    #         name="stair_s_u",
    #         dim=dim,
    #         rotations=(90, 180, 270),
    #         flips=("x", "y"),
    #         weight=1.0,
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
    #             ),
    #         ),
    #     ),
    # )
    # platform0: MeshPartsCfg = PlatformMeshPartsCfg(
    #     name="platform_1000",
    #     dim=dim,
    #     array=np.array([[1, 0], [0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # )
    # PlatformMeshPartsCfg(
    #     name="platform_1100",
    #     dim=dim,
    #     array=np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    #     # array=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_1110",
    #     dim=dim,
    #     array=np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]),
    #     # array=np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.10,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_1111",
    #     dim=dim,
    #     array=np.array([[1, 1], [1, 1]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_1111_f",
    #     dim=dim,
    #     array=np.array([[1, 1], [1, 1]]),
    #     z_dim_array=np.ones((2, 2)) * 0.1,
    #     use_z_dim_array=True,
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # platform4: MeshPartsCfg = PlatformMeshPartsCfg(
    #     name="platform_2000",
    #     dim=dim,
    #     array=np.array([[2, 0], [0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # )
    # PlatformMeshPartsCfg(
    #     name="platform_2200",
    #     dim=dim,
    #     array=np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.6,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2220",
    #     dim=dim,
    #     array=np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 0, 0, 0], [2, 2, 0, 0, 0], [2, 2, 0, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.60,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2222",
    #     dim=dim,
    #     array=np.array([[2, 2], [2, 2]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.6,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2211",
    #     dim=dim,
    #     array=np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.60,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2221",
    #     dim=dim,
    #     array=np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [2, 2, 1, 1, 1]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.60,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2111",
    #     dim=dim,
    #     array=np.array([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.60,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2111_f",
    #     dim=dim,
    #     array=np.array([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     z_dim_array=np.ones((5, 5)) * 0.1,
    #     use_z_dim_array=True,
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.60,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_2222_f",
    #     dim=dim,
    #     array=np.array([[2, 2], [2, 2]]),
    #     z_dim_array=np.ones((2, 2)) * 0.1,
    #     use_z_dim_array=True,
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.6,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_T",
    #     dim=dim,
    #     array=np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_T_f",
    #     dim=dim,
    #     array=np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
    #     z_dim_array=np.ones((5, 5)) * 0.5,
    #     use_z_dim_array=True,
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_stepping",
    #     dim=dim,
    #     array=np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_stepping_f",
    #     dim=dim,
    #     array=np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     z_dim_array=np.ones((5, 5)) * 0.5,
    #     use_z_dim_array=True,
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
    # PlatformMeshPartsCfg(
    #     name="platform_stepping",
    #     dim=dim,
    #     array=np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # ),
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
