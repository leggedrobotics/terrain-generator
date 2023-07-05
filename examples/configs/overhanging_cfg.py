import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from itertools import product

from numpy.random import f

from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    OverhangingMeshPartsCfg,
    OverhangingBoxesPartsCfg,
    MeshPattern,
    MeshPartsCfg,
    WallPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    FloatingBoxesPartsCfg,
)
from terrain_generator.trimesh_tiles.mesh_parts.rough_parts import generate_perlin_tile_configs

from terrain_generator.trimesh_tiles.patterns.pattern_generator import (
    generate_random_box_platform,
    # generate_walls,
    generate_floating_boxes,
    generate_narrow,
    generate_platforms,
    generate_ramp_parts,
    generate_stair_parts,
    generate_stepping_stones,
    generate_floating_capsules,
    generate_random_boxes,
    generate_overhanging_platforms,
    add_capsules,
)
from terrain_generator.trimesh_tiles.patterns.overhanging_patterns import generate_walls
from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile
from terrain_generator.trimesh_tiles.mesh_parts.basic_parts import create_from_height_map


@dataclass
class OverhangingTerrainPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234

    enable_wall: bool = False

    # random box platform
    random_cfgs = []
    n_random_boxes: int = 10
    random_box_weight: float = 0.01
    perlin_weight: float = 0.1
    for i in range(n_random_boxes):
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_{i}",
            offset=0.0,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_1.0{i}",
            offset=1.0,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_2.0{i}",
            offset=2.0,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_8_{i}",
            offset=0.0,
            height_diff=0.0,
            height_std=0.1,
            n=8,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        # random_cfgs += generate_random_box_platform(
        #     name=f"box_platform_flat_large_{i}",
        #     offset=0.0,
        #     height_diff=0.0,
        #     height_std=0.5,
        #     n=8,
        #     dim=dim,
        #     weight=random_box_weight / n_random_boxes,
        # )
        # random_cfgs += generate_random_box_platform(
        #     name=f"box_platform_0.0{i}",
        #     offset=0.0,
        #     height_diff=0.5,
        #     height_std=0.1,
        #     n=6,
        #     dim=dim,
        #     weight=random_box_weight / n_random_boxes,
        # )
        # random_cfgs += generate_random_box_platform(
        #     name=f"box_platform_0.5{i}",
        #     offset=0.5,
        #     height_diff=0.5,
        #     height_std=0.1,
        #     n=6,
        #     dim=dim,
        #     weight=random_box_weight / n_random_boxes,
        # )
        # random_cfgs += generate_random_box_platform(
        #     name=f"box_platform_1.0_{i}",
        #     offset=1.0,
        #     height_diff=0.5,
        #     height_std=0.1,
        #     n=6,
        #     dim=dim,
        #     weight=random_box_weight / n_random_boxes,
        # )
        # random_cfgs += generate_random_box_platform(
        #     name=f"box_platform_1.5_{i}",
        #     offset=1.5,
        #     height_diff=0.5,
        #     height_std=0.1,
        #     n=6,
        #     dim=dim,
        #     weight=random_box_weight / n_random_boxes,
        # )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_diff_1.0_{i}",
            offset=0.0,
            height_diff=1.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_diff_1.0_std{i}",
            offset=0.0,
            height_diff=1.0,
            height_std=0.15,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_diff_2.0_{i}",
            offset=1.0,
            height_diff=1.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_perlin_tile_configs(f"perlin_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes)
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_0.5_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=0.5, height=0.5
        )
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_1.0_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=0.0, height=1.0
        )
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_1.0_1.0{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=1.0, height=1.0
        )
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        (WallPartsCfg(name=f"floor", dim=dim, wall_edges=(), weight=0.01),)
        + tuple(
            generate_platforms(name="platform_1", dim=dim, max_h=1.0, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2", dim=dim, max_h=2.0, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2_1", dim=dim, max_h=2.0, min_h=1.0, weight=0.5, enable_wall=enable_wall)
        )
        # + tuple(
        #     generate_platforms(name="platform_0.5", dim=dim, max_h=0.5, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        # )
        # + tuple(
        #     generate_platforms(
        #         name="platform_1_0.5", dim=dim, max_h=1.0, min_h=0.5, weight=0.5, enable_wall=enable_wall
        #     )
        # )
        + tuple(random_cfgs)
        + tuple(
            generate_stair_parts(
                name="stair", dim=dim, seed=seed, array_shape=[15, 15], weight=0.5, depth_num=2, enable_wall=enable_wall
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_offset",
                dim=dim,
                seed=seed,
                array_shape=[15, 15],
                weight=2.0,
                depth_num=2,
                offset=1.0,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low",
                dim=dim,
                total_height=0.5,
                seed=seed,
                array_shape=[15, 15],
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
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
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
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
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
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
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
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
                weight=0.05,
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
                weight=0.2,
                depth_num=1,
            )
        )
    )


@dataclass
class OverhangingFloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234

    enable_wall: bool = False
    random_cfgs = []
    n_random_boxes: int = 10
    random_box_weight: float = 0.01
    perlin_weight: float = 0.1
    for i in range(n_random_boxes):
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_{i}",
            offset=0.0,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_0.2{i}",
            offset=0.2,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_0.3{i}",
            offset=0.3,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        (WallPartsCfg(name=f"floor", dim=dim, wall_edges=(), weight=0.01),)
        + tuple(
            generate_platforms(name="platform_1", dim=dim, max_h=0.2, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2", dim=dim, max_h=0.3, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2_1", dim=dim, max_h=0.3, min_h=0.2, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(random_cfgs)
    )


@dataclass
class OverhangingPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    mesh_parts: Tuple[MeshPartsCfg, ...] = generate_walls(name="walls", dim=dim, wall_height=3.0, wall_thickness=0.4)
    overhanging_prob: float = 0.5
    gap_means = [0.6, 0.8, 1.0, 1.2]
    gap_std = [0.05, 0.1, 0.1, 0.2]
    box_height = [0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    box_grid_n = [3, 4, 6]
    overhanging_cfgs = [
        OverhangingBoxesPartsCfg(gap_mean=gap_mean, gap_std=gap_std, box_height=box_height, box_grid_n=box_grid_n)
        for gap_mean, gap_std, box_height, box_grid_n in product(gap_means, gap_std, box_height, box_grid_n)
    ]
    box_dim_mins = [[0.5, 0.5, 0.1], [1.5, 0.05, 0.05]]
    box_dim_maxs = [[1.0, 1.0, 0.5], [1.5, 0.1, 0.1]]
    min_heights = [0.5, 0.6, 0.7, 1.0]
    max_heights = [1.0, 1.0, 1.0, 2.0]
    roll_pitch_range = [[0.0, np.pi / 6], [0.0, np.pi / 4], [0.0, np.pi / 2]]
    n_boxes = [3, 5]
    floating_box_cfgs = [
        FloatingBoxesPartsCfg(
            name="floating_boxes",
            n_boxes=n_box,
            box_dim_min=tuple(box_dim_min),
            box_dim_max=tuple(box_dim_max),
            roll_pitch_range=(0.0, np.pi / 6),  # in rad
            min_height=min_height,
        )
        for n_box, box_dim_min, box_dim_max, min_height in product(n_boxes, box_dim_mins, box_dim_maxs, min_heights)
    ]
    # overhanging_cfg_list: Tuple[OverhangingMeshPartsCfg, ...] = tuple(overhanging_cfgs) + tuple(floating_box_cfgs)
    overhanging_cfg_list: Tuple[OverhangingMeshPartsCfg, ...] = tuple(floating_box_cfgs)


if __name__ == "__main__":
    from utils import get_height_array_of_mesh

    cfg = OverhangingPattern()
    print(cfg)
    # print(len(cfg.overhanging_cfg_list))

    # exit(0)

    cfg = OverhangingTerrainPattern()
    # print(cfg)
    keywords = ["mesh"]
    for mesh_part in cfg.mesh_parts:
        print("name ", mesh_part.name)
        if any([keyword in mesh_part.name for keyword in keywords]):
            print(mesh_part)
            tile = create_mesh_tile(mesh_part)
            print("tile ", tile)
            mesh = tile.get_mesh()
            print("mesh ", mesh)
            mesh.show()
            print(get_height_array_of_mesh(mesh, mesh_part.dim, 5))
