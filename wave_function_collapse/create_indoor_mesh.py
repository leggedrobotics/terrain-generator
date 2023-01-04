import numpy as np
from typing import Tuple
import functools

from wfc.tiles import Tile, ArrayTile, MeshTile, MeshGeneratorTile
from mesh_parts.indoor_parts import create_wall_mesh, create_stairs_mesh, create_platform_mesh
from mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    WallMeshPartsCfg,
    MeshPattern,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
)
from mesh_parts.mesh_utils import get_height_array_of_mesh


def create_mesh_tile(cfg: MeshPartsCfg):
    if isinstance(cfg, WallMeshPartsCfg):
        mesh_gen = functools.partial(create_wall_mesh, cfg)
    elif isinstance(cfg, StairMeshPartsCfg):
        mesh_gen = functools.partial(create_stairs_mesh, cfg)
    elif isinstance(cfg, PlatformMeshPartsCfg):
        mesh_gen = functools.partial(create_platform_mesh, cfg)
    else:
        return
    name = cfg.name
    mesh = mesh_gen()
    if cfg.use_generator:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshGeneratorTile(name, array, mesh_gen, weight=cfg.weight)
    else:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, mesh, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern):
    tiles = []
    for mesh_cfg in cfg.mesh_parts:
        tile = create_mesh_tile(mesh_cfg)
        if tile is not None:
            tiles += tile.get_all_tiles(rotations=mesh_cfg.rotations, flips=mesh_cfg.flips)
    tile_dict = {tile.name: tile for tile in tiles}
    return tile_dict
