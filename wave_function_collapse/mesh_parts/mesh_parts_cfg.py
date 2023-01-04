import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeshPartsCfg:
    name: str = "mesh"
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: Tuple[int, ...] = ()  # (90, 180, 270)
    flips: Tuple[str, ...] = ()  # ("x", "y")
    height_offset: float = 0.0
    use_generator: bool = False


@dataclass
class WallMeshPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.1
    wall_height: float = 2.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right, middle_left, middle_right, middle_up, middle_bottom
    wall_type: str = "wall"  # wall, window, door
    # wall_type_probs: Tuple[float, ...] = (0.6, 0.2, 0.2)  # wall, window, door
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5
    door_direction: str = ""  # left, right, up, down, none
    # wall_array: np.ndarray = np.zeros((3, 3))


@dataclass
class StairMeshPartsCfg(MeshPartsCfg):
    @dataclass
    class Stair(MeshPartsCfg):
        step_width: float = 1.0
        step_depth: float = 0.3
        n_steps: int = 5
        total_height: float = 1.0
        height_offset: float = 0.0
        stair_type: str = "standard"  # stair, open, ramp
        add_residual_side_up: bool = True  # If false, add to bottom.
        add_rail: bool = False
        direction: str = "up"
        attach_side: str = "left"

    stairs: Tuple[Stair, ...] = (Stair(),)
    wall: Optional[WallMeshPartsCfg] = None


@dataclass
class PlatformMeshPartsCfg(MeshPartsCfg):
    array: np.ndarray = np.zeros((2, 2))
    add_floor: bool = True


@dataclass
class MeshPattern:
    name: str
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z


@dataclass
class FloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    floor: MeshPartsCfg = WallMeshPartsCfg(name="floor", dim=dim, wall_edges=(), weight=10.0)
    wall_straight: MeshPartsCfg = WallMeshPartsCfg(
        name="wall_s",
        dim=dim,
        wall_edges=("middle_left", "middle_right"),
        rotations=(90, 180, 270),
        flips=(),
        weight=2.0,
        door_direction="up",
    )
    wall_turn: MeshPartsCfg = WallMeshPartsCfg(
        name="wall_t",
        dim=dim,
        wall_edges=("middle_left", "middle_bottom"),
        rotations=(90, 180, 270),
        flips=(),
        weight=1.0,
        door_direction="",
    )
    wall_straight_door: MeshPartsCfg = WallMeshPartsCfg(
        name="door_s",
        dim=dim,
        wall_edges=("middle_left", "middle_right"),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.2,
        door_direction="up",
        create_door=True,
    )
    wall_straight_edge: MeshPartsCfg = WallMeshPartsCfg(
        name="wall_s_e",
        dim=dim,
        wall_edges=("left",),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        door_direction="",
    )

    # platform0: MeshPartsCfg = PlatformMeshPartsCfg(
    #     name="platform_1000",
    #     dim=dim,
    #     array=np.array([[1, 0], [0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # )

    platform1: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_1100",
        dim=dim,
        array=np.array([[1, 1], [0, 0]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
    )

    platform2: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_1110",
        dim=dim,
        array=np.array([[1, 1], [1, 0]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.10,
    )

    platform3: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_1111",
        dim=dim,
        array=np.array([[1, 1], [1, 1]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
    )

    # platform4: MeshPartsCfg = PlatformMeshPartsCfg(
    #     name="platform_2000",
    #     dim=dim,
    #     array=np.array([[2, 0], [0, 0]]),
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    # )

    platform5: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_2200",
        dim=dim,
        array=np.array([[2, 2], [0, 0]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
    )

    platform6: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_2220",
        dim=dim,
        array=np.array([[2, 2], [2, 0]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.10,
    )

    platform7: MeshPartsCfg = PlatformMeshPartsCfg(
        name="platform_2222",
        dim=dim,
        array=np.array([[2, 2], [2, 2]]),
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
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


@dataclass
class StairsPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    # floor_middle: MeshPartsCfg = WallMeshPartsCfg(
    #     name="floor_middle", dim=dim, wall_edges=(), weight=0.1, height_offset=1.0
    # )
    # floor_top: MeshPartsCfg = WallMeshPartsCfg(name="floor_top", dim=dim, wall_edges=(), weight=0.1, height_offset=2.0)
    #
    stair_straight: MeshPartsCfg = StairMeshPartsCfg(
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
    )
    stair_straight_up: MeshPartsCfg = StairMeshPartsCfg(
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
    )
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

    stair_wide: MeshPartsCfg = StairMeshPartsCfg(
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
    )
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
