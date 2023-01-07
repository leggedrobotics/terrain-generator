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


def generate_platforms(name, dim, max_h=1.0, min_h=0.0, weight=1.0, seed=1234):
    platform_types = ["1100", "1110", "1111", "1111_f"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "1100":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1110":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1111":
            array = np.ones((5, 5))
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1111_f":
            array = np.ones((5, 5))
            array = min_h + array * (max_h - min_h)
            z_dim_array = np.ones((5, 5)) * 0.1
            use_z_dim_array = True

        wall_patterns = [
            (),
            ("middle_left", "middle_right"),
            ("middle_bottom", "middle_right"),
            # ("middle_left", "middle_bottom"),
        ]
        weights = [
            weight,
            weight * 0.1,
            weight * 0.1,
        ]
        for i, wall_pattern in enumerate(wall_patterns):
            new_name = f"{name}_{platform_type}"
            if len(wall_pattern) > 0:
                new_name += f"_wall_{wall_pattern}"
            cfg = PlatformMeshPartsCfg(
                name=new_name,
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weights[i],
                use_z_dim_array=use_z_dim_array,
                wall=WallMeshPartsCfg(dim=dim, wall_edges=wall_pattern),
            )
            cfgs.append(cfg)
    return cfgs


def generate_narrow(name, dim, max_h=1.0, min_h=0.0, weight=1.0, seed=1234):
    platform_types = ["I", "T", "T2", "L", "L2", "TT", "S", "S2", "PI", "PL"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "I":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "T":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "T":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "L":
            array = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        elif platform_type == "L2":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        elif platform_type == "TT":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]])
        elif platform_type == "PI":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif platform_type == "PL":
            array = np.array(
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif max_h - min_h <= 1.0 and platform_type == "S":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif max_h - min_h <= 1.0 and platform_type == "S2":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 1],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        else:
            continue
        array = min_h + array * (max_h - min_h)
        z_dim_array = array

        n = len(platform_types) * 2.0
        weight_per_tile = weight / n
        for prefix in ["", "_f"]:
            if prefix == "_f":
                z_dim_array = np.ones(array.shape) * 0.1
                use_z_dim_array = True
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_{platform_type}{prefix}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weight_per_tile,
                use_z_dim_array=use_z_dim_array,
            )
            cfgs.append(cfg)
    return cfgs


def generate_stepping_stones(name, dim, max_h=1.0, min_h=0.0, weight=1.0, seed=1234):
    platform_types = ["1100", "1110", "1111", "s", "s2", "p", "p2"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "1100":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        elif platform_type == "1110":
            array = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "1111":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "s":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "s2":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "p":
            array = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]])
        array = min_h + array * (max_h - min_h)
        z_dim_array = array

        for prefix in ["", "_f"]:
            if prefix == "_f":
                z_dim_array = np.ones((5, 5)) * 0.1
                use_z_dim_array = True
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_{platform_type}{prefix}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weight,
                use_z_dim_array=use_z_dim_array,
            )
            cfgs.append(cfg)
    return cfgs


def generate_floating_boxes(dim, n=15, array_shape=[5, 5], weight=1.0, seed=1234):
    np.random.seed(seed)
    weight_per_tile = weight / n
    cfgs = []
    for i in range(n):
        name = f"random_floating_box_{i}"
        # Randomly sample array
        array = np.random.uniform(0, 1, size=array_shape)
        array = np.round(array, 2)
        array[0, :] = np.round(array[0, :], 1)
        array[-1, :] = np.round(array[-1, :], 1)
        array[:, 0] = np.round(array[:, 0], 1)
        array[:, -1] = np.round(array[:, -1], 1)
        array = array.clip(0.1, 1.0)

        # Randomly create edges
        if np.random.uniform(0, 1) < 0.5:
            array[0, :] = 0.0
        if np.random.uniform(0, 1) < 0.5:
            array[-1, :] = np.array([1, 1, 1, 0, 0])
            # array[-1, :] = 1.0
        if np.random.uniform(0, 1) < 0.2:
            array[:, 0] = np.array([1, 1, 1, 0, 0])
        if np.random.uniform(0, 1) < 0.2:
            array[:, 0] = 0
        if np.random.uniform(0, 1) < 0.2:
            array[:, -1] = 0

        # Randomly sample z_dim_array
        z_dim_array = np.random.uniform(0.0, 1, size=array_shape) * array
        z_dim_array = z_dim_array.clip(0.0, array)
        z_dim_array = z_dim_array.clip(0.1, 1.0)
        z_dim_array = np.round(z_dim_array, 2)

        # print("array ", array)
        # print("z_dim_array ", z_dim_array)
        cfg = PlatformMeshPartsCfg(
            name=name,
            dim=dim,
            array=array,
            z_dim_array=z_dim_array,
            use_z_dim_array=True,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
        )
        # print("cfg ", cfg)
        cfgs.append(cfg)
    return cfgs


def generate_stair_parts(
    name,
    dim,
    total_height=1.0,
    step_height=0.2,
    array_shape=[10, 10],
    offset=0.0,
    step_thickness=0.1,
    depth_num=1,
    weight=1.0,
    seed=1234,
):
    np.random.seed(seed)
    arrays = []
    z_dim_arrays = []
    step_height = max(step_height, total_height / array_shape[1])
    n_steps = int(total_height // step_height)
    # residual = total_height - step_height * n_steps
    stair_types = ["wide", "half", "wide_float", "half_float", "corner", "corner_flipped", "turn"]
    wall_patterns = [
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [()],
        [()],
        [(), ("middle_left", "middle_bottom"), ("up", "right")],
    ]
    for stair_type in stair_types:
        h_1 = 0.0 + offset
        h_2 = total_height + offset
        # Randomly sample array
        array_1 = np.zeros(array_shape) + offset  # stairs aligned to up
        array_2 = np.zeros(array_shape) + offset  # stairs aligned to down
        z_dim_array_1 = np.zeros(array_shape)
        z_dim_array_2 = np.zeros(array_shape)
        for s in range(array_shape[0]):
            # print("s ", s)
            # print("h1 ", h_1, " h2 ", h_2)
            if stair_type == "wide":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
                z_dim_array_1[:, s] = h_1
                z_dim_array_2[:, s] = h_2
            elif stair_type == "half":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
                z_dim_array_1[: int(array_shape[0] / 2), s] = h_1
                z_dim_array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "wide_float":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
                z_dim_array_1[:, s] = 0.1
                z_dim_array_2[:, s] = 0.1
            elif stair_type == "half_float":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
                z_dim_array_1[: int(array_shape[0] / 2), s] = step_thickness
                z_dim_array_2[: int(array_shape[0] / 2), s] = step_thickness
            elif stair_type == "corner":
                array_1[:s, s] = h_1
                array_2[:s, s] = h_2
                array_1[s, :s] = h_1
                array_2[s, :s] = h_2
                array_1[s, s] = h_1
                array_2[s, s] = h_2
                z_dim_array_1 = array_1
                z_dim_array_2 = array_2
            elif stair_type == "corner_flipped":
                array_1[:s, s] = total_height - h_1 + offset
                array_2[:s, s] = total_height - h_2 + offset
                array_1[s, :s] = total_height - h_1 + offset
                array_2[s, :s] = total_height - h_2 + offset
                array_1[s, s] = total_height - h_1 + offset
                array_2[s, s] = total_height - h_2 + offset
                z_dim_array_1 = array_1
                z_dim_array_2 = array_2
            elif stair_type == "turn":
                half_idx = int(array_shape[0] / 2)
                if s < half_idx:
                    array_1[:half_idx, s] = h_1
                    array_2[:half_idx, s] = h_2
                    z_dim_array_1[:half_idx, s] = h_1
                    z_dim_array_2[:half_idx, s] = h_2
                else:
                    array_1[s, half_idx:] = h_1
                    array_2[s, half_idx:] = h_2
                    z_dim_array_1[s, half_idx:] = h_1
                    z_dim_array_2[s, half_idx:] = h_2
                if s == half_idx:
                    array_1[:half_idx, half_idx:] = h_1
                    array_2[:half_idx, half_idx:] = h_2
                    z_dim_array_1[:half_idx, half_idx:] = h_1
                    z_dim_array_2[:half_idx, half_idx:] = h_2
            if s % depth_num == 0 and s > 0:
                h_1 = min(h_1 + step_height, total_height + offset)
                h_2 = max(h_2 - step_height, 0.0 + offset)

        # print(stair_type)
        # print("array 1 \n", array_1)
        # print("array 2 \n", array_2)
        arrays.append(np.round(array_1, 1))
        arrays.append(np.round(array_2, 1))
        z_dim_arrays.append(np.round(z_dim_array_1, 1))
        z_dim_arrays.append(np.round(z_dim_array_2, 1))

    weight_per_tile = weight / len(arrays)
    # weights = [weight_per_tile, weight_per_tile * 0.5]
    wall_weight = weight_per_tile * 0.5
    # print("weight_per_tile", weight_per_tile)
    cfgs = []
    for i, (array, z_dim_array) in enumerate(zip(arrays, z_dim_arrays)):
        for j, wall_pattern in enumerate(wall_patterns[i // 2]):
            weight = weight_per_tile
            if len(wall_pattern) > 0:
                weight = wall_weight
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_wall_{wall_pattern}_{i}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                use_z_dim_array=True,
                rotations=(90, 180, 270),
                flips=("x", "y"),
                weight=weight,
                wall=WallMeshPartsCfg(
                    dim=dim,
                    wall_edges=wall_pattern,
                ),
            )
            cfgs.append(cfg)
    return cfgs


def generate_ramp_parts(
    name,
    dim,
    total_height=1.0,
    array_shape=[10, 10],
    offset=0.0,
    depth_num=1,
    weight=1.0,
    seed=1234,
):
    np.random.seed(seed)
    arrays = []
    step_height = total_height / (array_shape[1] - 2)
    cfg = HeightMapMeshPartsCfg()
    # residual = total_height - step_height * n_steps
    ramp_types = ["wide", "half", "wide_float", "half_float", "corner", "corner_flipped"]
    for stair_type in ramp_types:
        h_1 = 0.0 + offset
        h_2 = total_height + offset
        # Randomly sample array
        array_1 = np.zeros(array_shape) + offset  # stairs aligned to up
        array_2 = np.zeros(array_shape) + offset  # stairs aligned to down
        for s in range(array_shape[0]):
            # print("s ", s)
            # print("h1 ", h_1, " h2 ", h_2)
            if stair_type == "wide":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
            elif stair_type == "half":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "wide_float":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
            elif stair_type == "half_float":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "corner":
                array_1[:s, s] = h_1
                array_2[:s, s] = h_2
                array_1[s, :s] = h_1
                array_2[s, :s] = h_2
                array_1[s, s] = h_1
                array_2[s, s] = h_2
            elif stair_type == "corner_flipped":
                array_1[:s, s] = total_height - h_1 + offset
                array_2[:s, s] = total_height - h_2 + offset
                array_1[s, :s] = total_height - h_1 + offset
                array_2[s, :s] = total_height - h_2 + offset
                array_1[s, s] = total_height - h_1 + offset
                array_2[s, s] = total_height - h_2 + offset
            if s % depth_num == 0 and s > 0:
                h_1 = min(h_1 + step_height, total_height + offset)
                h_2 = max(h_2 - step_height, offset)

        # print("array 1 \n", array_1)
        # print("array 2 \n", array_2)
        arrays.append(array_1)
        arrays.append(array_2)

    weight_per_tile = weight / len(ramp_types) / 2
    # print("weight_per_tile", weight_per_tile)
    cfgs = []
    for i, array in enumerate(arrays):
        cfg = HeightMapMeshPartsCfg(
            name=f"{name}_{ramp_types[i // 2]}_{i}",
            dim=dim,
            height_map=array,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
            slope_threshold=0.5,
            target_num_faces=1000,
            simplify=False,
        )
        cfgs.append(cfg)
    return cfgs


@dataclass
class FloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        (
            WallMeshPartsCfg(name="floor", dim=dim, wall_edges=(), weight=13.0),
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
            PlatformMeshPartsCfg(
                name="platform_stepping_rand",
                dim=dim,
                array=np.array(
                    [[0, 0, 1, 0, 0], [0, 0.8, 0, 0.6, 0], [1, 0, 0.7, 0, 0], [1, 0.9, 0, 0, 0], [1, 1, 1, 0, 0]]
                ),
                rotations=(90, 180, 270),
                flips=(),
                weight=0.1,
            ),
            PlatformMeshPartsCfg(
                name="platform_stepping_rand_2",
                dim=dim,
                array=np.array(
                    [
                        [0, 0, 0, 1, 1],
                        [0, 0.0, 0.7, 0.6, 0],
                        [0, 0.4, 0.5, 0, 0],
                        [0, 0.2, 0.1, 0.3, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                rotations=(90, 180, 270),
                flips=(),
                weight=0.1,
            ),
            PlatformMeshPartsCfg(
                name="platform_stepping_rand_3",
                dim=dim,
                array=np.array(
                    [
                        [1, 1, 0, 0, 0],
                        [0, 0.0, 0.7, 0.6, 0],
                        [0, 0.4, 0.5, 0, 0],
                        [0, 0.2, 0.1, 0.3, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                rotations=(90, 180, 270),
                flips=(),
                weight=0.1,
            ),
            # PlatformMeshPartsCfg(
            #     name="platform_2.0",
            #     dim=dim,
            #     array=np.ones((5, 5)) * 2.0,
            #     ),
            #     rotations=(90, 180, 270),
            #     flips=(),
            #     weight=0.1,
            # ),
        )
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


if __name__ == "__main__":
    cfg = FloorPattern()
    # print("cfg", cfg)
    from mesh_parts.create_tiles import create_mesh_tile

    visualize_keywords = ["stair_wall_()_12", ")_12"]
    for mesh_part in cfg.mesh_parts:
        for keyword in visualize_keywords:
            # print(mesh_part.edges)
            if keyword in mesh_part.name:
                mesh_tile = create_mesh_tile(mesh_part)
                print(mesh_tile.name, mesh_tile.edges)
                mesh_tile.get_mesh().show()
                break

        # if "ramp" in mesh_part.name:
        #     print(mesh_part)
        #     mesh_tile = create_mesh_tile(mesh_part)
        #     mesh_tile.get_mesh().show()
