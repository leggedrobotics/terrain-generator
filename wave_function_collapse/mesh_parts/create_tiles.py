import numpy as np
from typing import Tuple, Callable
import trimesh
import functools

from wfc.tiles import Tile, ArrayTile, MeshTile
from mesh_parts.indoor_parts import create_stairs_mesh
from mesh_parts.basic_parts import (
    create_floor,
    create_platform_mesh,
    create_from_height_map,
    create_wall_mesh,
    create_capsule_mesh,
    create_box_mesh,
)
from mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    WallMeshPartsCfg,
    MeshPattern,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    CapsuleMeshPartsCfg,
    BoxMeshPartsCfg,
    CombinedMeshPartsCfg,
)
from mesh_parts.mesh_utils import get_height_array_of_mesh, get_cached_mesh_gen, merge_meshes
from alive_progress import alive_it


def get_mesh_gen(cfg: MeshPartsCfg) -> Callable:
    if isinstance(cfg, WallMeshPartsCfg):
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
    elif isinstance(cfg, CombinedMeshPartsCfg):
        # print("getting CombinedMeshPartsCfg gen")
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
    if cfg.use_generator:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, cached_mesh_gen, mesh_dim=cfg.dim, weight=cfg.weight)
    else:
        array = get_height_array_of_mesh(mesh, cfg.dim, 5)
        return MeshTile(name, array, mesh, mesh_dim=cfg.dim, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern):
    import ray

    ray.init(ignore_reinit_error=True)
    create_mesh_tile_remote = ray.remote(create_mesh_tile)

    tiles = []
    print("Creating mesh pattern... ")
    for mesh_cfg in alive_it(cfg.mesh_parts):
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
