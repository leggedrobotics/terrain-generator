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
        # step_height: float = 0.2
        step_depth: float = 0.3
        n_steps: int = 5
        total_height: float = 1.0
        height_offset: float = 0.0
        # stair_direction: str = ""
        stair_type: str = "standard"  # stair, open, ramp
        add_residual_side_up: bool = True  # If false, add to bottom.
        add_rail: bool = False
        fill_bottom: bool = False
        direction: str = "up"
        gap_direction: str = "up"
        start_offset: float = 0.0
        attach_side: str = "left"

    stairs: Tuple[Stair, ...] = (Stair(),)
    wall: Optional[WallMeshPartsCfg] = None


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
    # wall_straight: MeshPartsCfg = WallMeshPartsCfg(name="wall_s", dim=dim, wall_edges=("up",), rotations=(90, 180, 270), flips=(), weight=2.0)
    # wall_turn: MeshPartsCfg = WallMeshPartsCfg(name="wall_t", dim=dim, wall_edges=("up", "right"), rotations=(90, 180, 270), flips=(), weight=1.0)
    # wall_straight_door: MeshPartsCfg = WallMeshPartsCfg(name="door_s", dim=dim, wall_edges=("up",), rotations=(90, 180, 270), flips=(), weight=0.2, door_direction="up", create_door=True)


@dataclass
class StairPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    floor: MeshPartsCfg = WallMeshPartsCfg(name="floor", dim=dim, wall_edges=(), weight=10.0)
    stair_straight: MeshPartsCfg = StairMeshPartsCfg(
        name="stair_s",
        dim=dim,
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=1.0,
                # step_height=0.2,
                step_depth=0.4,
                total_height=1.0,
                stair_type="standard",
                direction="up",
                gap_direction="up",  # up means gap after going up the stairs. down means gap before going up the stairs
                attach_side="up_right_front",
                add_rail=False,
                fill_bottom=False,
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
