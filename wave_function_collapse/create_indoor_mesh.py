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


# def create_wall_meshtile(cfg: WallMeshPartsCfg):
#     # array = np.zeros((3, 3))
#     name = cfg.name
#     for edge in cfg.wall_edges:
#         name += f"_{edge}"
#     if cfg.use_generator:
#         mesh_gen = functools.partial(create_wall_mesh, cfg)
#         mesh = mesh_gen()
#         array = get_height_array_of_mesh(mesh, cfg.dim, 5)
#         return MeshGeneratorTile(name, array, mesh_gen, weight=cfg.weight)
#     else:
#         mesh = create_wall_mesh(cfg)
#         array = get_height_array_of_mesh(mesh, cfg.dim, 5)
#         return MeshTile(name, array, mesh, weight=cfg.weight)
#
#
# def create_stair_meshtile(cfg: StairMeshPartsCfg):
#     # array = np.zeros((3, 3))
#     name = cfg.name
#     # for edge in cfg.wall_edges:
#     #     name += f"_{edge}"
#     if cfg.use_generator:
#         mesh_gen = functools.partial(create_stairs_mesh, cfg)
#         mesh = mesh_gen()
#         array = get_height_array_of_mesh(mesh, cfg.dim, 5)
#         return MeshGeneratorTile(name, array, mesh_gen, weight=cfg.weight)
#     else:
#         mesh = create_stairs_mesh(cfg)
#         array = get_height_array_of_mesh(mesh, cfg.dim, 5)
#         return MeshTile(name, array, mesh, weight=cfg.weight)


def create_mesh_tile(cfg: MeshPartsCfg):
    if isinstance(cfg, WallMeshPartsCfg):
        mesh_gen = functools.partial(create_wall_mesh, cfg)
    elif isinstance(cfg, StairMeshPartsCfg):
        mesh_gen = functools.partial(create_stairs_mesh, cfg)
    elif isinstance(cfg, PlatformMeshPartsCfg):
        mesh_gen = functools.partial(create_platform_mesh, cfg)
    else:
        return
    # array = np.zeros((3, 3))
    name = cfg.name
    mesh = mesh_gen()
    # for edge in cfg.wall_edges:
    #     name += f"_{edge}"
    if cfg.use_generator:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshGeneratorTile(name, array, mesh_gen, weight=cfg.weight)
    else:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, mesh, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern):
    tiles = []
    for k, v in cfg.__dict__.items():
        tile = create_mesh_tile(v)
        if tile is not None:
            tiles += tile.get_all_tiles(rotations=v.rotations, flips=v.flips)

        # if isinstance(v, WallMeshPartsCfg):
        #     tiles += create_wall_meshtile(v).get_all_tiles(rotations=v.rotations, flips=v.flips)
        # if isinstance(v, StairMeshPartsCfg):
        #     tiles += create_stair_meshtile(v).get_all_tiles(rotations=v.rotations, flips=v.flips)
        # if isinstance(v, PlatformMeshPartsCfg):
        #     tiles += create_platform_meshtile(v).get_all_tiles(rotations=v.rotations, flips=v.flips)
    tile_dict = {tile.name: tile for tile in tiles}
    return tile_dict


# def create_wall_tiles(dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)):
#     tiles = []
#     cfg = WallMeshPartsCfg(dim=dim, wall_edges=())
#     tiles += create_wall_meshtile(cfg).get_all_tiles()
#
#     # cfg = WallMeshPartsCfg(dim=dim, wall_edges=("left",))
#     # tiles += create_wall_meshtile(cfg).get_all_tiles(rotations=(90, 180, 270), flips=())
#     #
#     # cfg = WallMeshPartsCfg(dim=dim, wall_edges=("left", "up"))
#     # tiles += create_wall_meshtile(cfg).get_all_tiles(rotations=(90, 180, 270), flips=())
#
#     cfg = WallMeshPartsCfg(dim=dim, wall_edges=("middle_left", "middle_right"))
#     tiles += create_wall_meshtile(cfg).get_all_tiles(rotations=(90, 180, 270), flips=())
#
#     # cfg = WallMeshPartsCfg(dim=dim, wall_edges=("left", "up"))
#     cfg = WallMeshPartsCfg(dim=dim, wall_edges=("middle_left", "middle_bottom"))
#     tiles += create_wall_meshtile(cfg).get_all_tiles(rotations=(90, 180, 270), flips=())
#
#     tile_dict = {tile.name: tile for tile in tiles}
#
#     # cfg = WallMeshPartsCfg(wall_edges=("left", "up", "right"))
#     # tiles += create_wall_meshtile(cfg).get_all_tiles(rotations=(90, 180, 270), flips=())
#
#     return tile_dict
