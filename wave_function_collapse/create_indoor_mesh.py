import numpy as np
from typing import Tuple
import functools

from wfc.tiles import Tile, ArrayTile, MeshTile
from mesh_parts.indoor_parts import create_wall_mesh, create_stairs_mesh, create_platform_mesh
from mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    WallMeshPartsCfg,
    MeshPattern,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
)
from mesh_parts.mesh_utils import get_height_array_of_mesh, get_cached_mesh_gen


def create_mesh_tile(cfg: MeshPartsCfg):
    if isinstance(cfg, WallMeshPartsCfg):
        mesh_gen = create_wall_mesh
        # mesh_gen = functools.partial(create_wall_mesh, cfg)
    elif isinstance(cfg, StairMeshPartsCfg):
        mesh_gen = create_stairs_mesh
        # mesh_gen = functools.partial(create_stairs_mesh, cfg)
    elif isinstance(cfg, PlatformMeshPartsCfg):
        mesh_gen = create_platform_mesh
        # mesh_gen = functools.partial(create_platform_mesh, cfg)
    else:
        return
    cached_mesh_gen = get_cached_mesh_gen(mesh_gen, cfg, verbose=True)
    name = cfg.name
    mesh = cached_mesh_gen()
    if cfg.use_generator:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, cached_mesh_gen, mesh_dim=cfg.dim, weight=cfg.weight)
    else:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, mesh, mesh_dim=cfg.dim, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern):
    tiles = []
    for mesh_cfg in cfg.mesh_parts:
        tile = create_mesh_tile(mesh_cfg)
        if tile is not None:
            tiles += tile.get_all_tiles(rotations=mesh_cfg.rotations, flips=mesh_cfg.flips)
    tile_dict = {tile.name: tile for tile in tiles}
    return tile_dict
