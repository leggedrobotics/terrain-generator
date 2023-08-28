#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
from typing import Tuple, Callable
import trimesh
import functools

from .indoor_parts import create_stairs_mesh
from .basic_parts import (
    create_floor,
    create_platform_mesh,
    create_from_height_map,
    create_wall_mesh,
    create_capsule_mesh,
    create_box_mesh,
)
from .overhanging_parts import generate_wall_from_array
from .mesh_parts_cfg import (
    MeshPartsCfg,
    WallPartsCfg,
    MeshPattern,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    CapsuleMeshPartsCfg,
    BoxMeshPartsCfg,
    CombinedMeshPartsCfg,
    WallMeshPartsCfg,
)
from ...wfc.tiles import MeshTile
from ...utils import get_height_array_of_mesh, get_cached_mesh_gen, merge_meshes

# from alive_progress import alive_it


def get_mesh_gen(cfg: MeshPartsCfg) -> Callable:
    if isinstance(cfg, WallPartsCfg):
        mesh_gen = create_wall_mesh
    elif isinstance(cfg, StairMeshPartsCfg):
        mesh_gen = create_stairs_mesh
    elif isinstance(cfg, PlatformMeshPartsCfg):
        mesh_gen = create_platform_mesh
    elif isinstance(cfg, HeightMapMeshPartsCfg):
        mesh_gen = create_from_height_map
    elif isinstance(cfg, CapsuleMeshPartsCfg):
        mesh_gen = create_capsule_mesh
    elif isinstance(cfg, BoxMeshPartsCfg):
        mesh_gen = create_box_mesh
    elif isinstance(cfg, WallMeshPartsCfg):
        mesh_gen = generate_wall_from_array
    elif isinstance(cfg, CombinedMeshPartsCfg):
        mesh_gens = [get_mesh_gen(c) for c in cfg.cfgs]

        def mesh_gen(cfg):
            # print("Generating CombinedMeshPartsCfg from mesh_gen")
            mesh = trimesh.Trimesh()
            for i, gen in enumerate(mesh_gens):
                # print("Generating mesh part ", i)
                # print("gen: ", gen)
                # print("cfg: ", cfg.cfgs[i])
                # print("mesh: ", mesh)
                new_mesh = gen(cfg.cfgs[i], mesh=mesh)
                # mesh = merge_meshes([mesh, new_mesh], cfg.minimal_triangles)
                mesh = merge_meshes([mesh, new_mesh], False)
            return mesh

        # print("defined mesh_gen", mesh_gen)

        # mesh_gen = lambda: functools.reduce(lambda a, b: a + b, [gen() for gen in mesh_gens])
    else:
        raise NotImplementedError(f"Mesh generator for {cfg} not implemented")
    return mesh_gen


# @ray.remote
def create_mesh_tile(cfg: MeshPartsCfg) -> MeshTile:
    mesh_gen = get_mesh_gen(cfg)
    cached_mesh_gen = get_cached_mesh_gen(mesh_gen, cfg, verbose=False, use_cache=cfg.load_from_cache)
    name = cfg.name
    mesh = cached_mesh_gen()
    # If edge array is not provided, create it from the mesh
    if cfg.edge_array is None:
        cfg.edge_array = get_height_array_of_mesh(mesh, cfg.dim, 5)
    # Create the tile
    if cfg.use_generator:
        return MeshTile(name, cached_mesh_gen, array=cfg.edge_array, mesh_dim=cfg.dim, weight=cfg.weight)
    else:
        return MeshTile(name, mesh, array=cfg.edge_array, mesh_dim=cfg.dim, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern) -> dict:
    import ray

    ray.init(ignore_reinit_error=True)
    create_mesh_tile_remote = ray.remote(create_mesh_tile)

    tiles = []
    print("Creating mesh pattern... ")
    for mesh_cfg in cfg.mesh_parts:
        tiles.append(create_mesh_tile_remote.remote(mesh_cfg))
    print("Waiting for parallel creation... ")
    tiles = ray.get(tiles)
    all_tiles = []
    for i, tile in enumerate(tiles):
        mesh_cfg = cfg.mesh_parts[i]
        all_tiles += tile.get_all_tiles(rotations=mesh_cfg.rotations, flips=mesh_cfg.flips)
        # tile = create_mesh_tile(mesh_cfg)
        # if tile is not None:
        #     tiles += tile.get_all_tiles(rotations=mesh_cfg.rotations, flips=mesh_cfg.flips)
    tile_dict = {tile.name: tile for tile in all_tiles}
    return tile_dict
