#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from numpy.random import f

from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    MeshPartsCfg,
    WallPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
)
from terrain_generator.trimesh_tiles.mesh_parts.rough_parts import generate_perlin_tile_configs

from terrain_generator.trimesh_tiles.patterns.pattern_generator import (
    generate_walls,
    generate_floating_boxes,
    generate_narrow,
    generate_platforms,
    generate_ramp_parts,
    generate_stair_parts,
    generate_stepping_stones,
)
from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile
from terrain_generator.trimesh_tiles.mesh_parts.basic_parts import create_from_height_map


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
        + tuple(
            generate_floating_boxes(name="floating_boxes", n=30, dim=dim, seed=seed, array_shape=[5, 5], weight=10.0)
        )
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


@dataclass
class IndoorPatternLevels(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234
    levels: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    wall_height: float = 0.5
    mesh_parts: Tuple[MeshPartsCfg, ...] = ()

    def __post_init__(self):
        cfgs = ()
        dim = self.dim
        seed = self.seed
        wall_height = self.wall_height
        min_hs = self.levels[:-1] + tuple(
            [self.levels[0], self.levels[2]]
        )  # + tuple([0.0 for _ in range(len(self.levels) - 2)])
        max_hs = self.levels[1:] + tuple([self.levels[2], self.levels[2 + 2]])  # + self.levels[2:]
        # for i in range(len(self.levels) - 2):
        for min_h, max_h in zip(min_hs, max_hs):
            # min_h = self.levels[i]
            # max_h = self.levels[i + 1]
            cfg = (
                tuple(generate_walls(dim, wall_height=wall_height))
                + tuple(
                    generate_platforms(
                        name=f"platform_{min_h}_{max_h}",
                        dim=dim,
                        max_h=max_h,
                        min_h=min_h,
                        weight=0.5,
                        wall_height=wall_height,
                    )
                )
                + tuple(
                    generate_stepping_stones(
                        name=f"stepping_{min_h}_{max_h}", dim=dim, max_h=max_h, min_h=min_h, weight=1.2
                    )
                )
                + tuple(generate_narrow(name=f"narrow_{min_h}_{max_h}", dim=dim, max_h=max_h, min_h=min_h, weight=0.2))
                + tuple(
                    generate_floating_boxes(
                        name=f"floating_boxes_{min_h}_{max_h}",
                        n=30,
                        dim=dim,
                        max_h=max_h,
                        min_h=min_h,
                        seed=seed,
                        array_shape=[5, 5],
                        weight=1.05,
                    )
                )
                + tuple(
                    generate_stair_parts(
                        name=f"stair_{min_h}_{max_h}",
                        dim=dim,
                        seed=seed,
                        array_shape=[15, 15],
                        weight=1.0,
                        depth_num=2,
                        total_height=max_h - min_h,
                        wall_height=wall_height,
                        offset=min_h,
                    )
                )
                + tuple(
                    generate_ramp_parts(
                        name=f"ramp_{min_h}_{max_h}",
                        dim=dim,
                        seed=seed,
                        array_shape=[30, 30],
                        total_height=max_h - min_h,
                        offset=min_h,
                        weight=1.0,
                        depth_num=1,
                    )
                )
            )
            cfgs += cfg
        self.mesh_parts = cfgs


if __name__ == "__main__":
    from terrain_generator.utils import get_height_array_of_mesh

    cfg = IndoorPatternLevels()
    # print(cfg)
    keywords = ["floating_boxes_0.0_1.0"]
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
