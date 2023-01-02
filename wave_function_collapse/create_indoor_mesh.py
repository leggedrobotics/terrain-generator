import numpy as np
from typing import Tuple

from wfc.tiles import Tile, ArrayTile, MeshTile
from mesh_parts.indoor_parts import create_wall_mesh
from mesh_parts.mesh_parts_cfg import MeshPartsCfg, WallMeshPartsCfg, MeshPattern


def create_wall_meshtile(cfg: WallMeshPartsCfg):
    mesh = create_wall_mesh(cfg)
    array = np.zeros((3, 3))
    name = "wall"
    for edge in cfg.wall_edges:
        if edge == "bottom":
            array[-1, :] = 1
        elif edge == "up":
            array[0, :] = 1
        elif edge == "left":
            array[:, 0] = 1
        elif edge == "right":
            array[:, -1] = 1
        elif edge == "middle_left":
            array[:, 0] = np.array([0, 1, 0])
        elif edge == "middle_right":
            array[:, -1] = np.array([0, 1, 0])
        elif edge == "middle_up":
            array[0, 0] = np.array([0, 1, 0])
        elif edge == "middle_bottom":
            array[-1, :] = np.array([0, 1, 0])
        else:
            raise ValueError(f"Edge {edge} is not defined.")
        name += f"_{edge}"
    return MeshTile(name, array, mesh, weight=cfg.weight)


def create_mesh_pattern(cfg: MeshPattern):
    tiles = []
    for k, v in cfg.__dict__.items():
        if isinstance(v, MeshPartsCfg):
            tiles += create_wall_meshtile(v).get_all_tiles(rotations=v.rotations, flips=v.flips)
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
