import trimesh
import numpy as np
from mesh_parts.mesh_parts_cfg import MeshPartsCfg, WallMeshPartsCfg


def merge_meshes(meshes, minimal_triangles=False):
    if minimal_triangles:
        meshes = trimesh.boolean.union(meshes, engine="blender")
    else:
        meshes = trimesh.util.concatenate(meshes)
    return meshes


def get_standard_wall(cfg: WallMeshPartsCfg, edge:str = "bottom"):
    if edge == "bottom":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [0, -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "up":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [0, cfg.dim[1] / 2.0 - cfg.wall_thickness / 2.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "left":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [-cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "right":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "middle_bottom":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
        pos = [0, -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "middle_up":
        dim = [cfg.wall_thickness, cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_height]
        pos = [0, cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "middle_left":
        dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [-cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif edge == "middle_right":
        dim = [cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, cfg.wall_thickness, cfg.wall_height]
        pos = [cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    else:
        raise ValueError(f"Edge {edge} is not defined.")

    pose = np.eye(4)
    pose[:3, -1] = pos
    wall = trimesh.creation.box(dim, pose)
    return wall


def get_door(cfg: WallMeshPartsCfg, door_direction:str = "up"):
    if door_direction == "bottom" or door_direction == "up":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction == "left" or door_direction == "right":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction == "middle_bottom":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [0, -cfg.dim[1] / 4.0 + cfg.wall_thickness / 4.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction == "middle_up":
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [0, cfg.dim[1] / 4.0 - cfg.wall_thickness / 4.0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif door_direction == "middle_left":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [-cfg.dim[0] / 4.0 + cfg.wall_thickness / 4.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    elif door_direction == "middle_right":
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [cfg.dim[0] / 4.0 - cfg.wall_thickness / 4.0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.wall_height / 2.0]
    else:
        return trimesh.Trimesh()

    pose = np.eye(4)
    pose[:3, -1] = pos
    door = trimesh.creation.box(dim, pose)
    return door


def create_wall_mesh(cfg: WallMeshPartsCfg):
    # Create the vertices of the wall
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0]
    floor = trimesh.creation.box(dims, pose)
    mesh = floor
    for wall_edges in cfg.wall_edges:
        wall = get_standard_wall(cfg, wall_edges)
        # wall = get_wall_with_door(cfg, wall_edges)
        mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)
    if cfg.create_door:
        door = get_door(cfg, cfg.door_direction)
        mesh = trimesh.boolean.difference([mesh, door], engine="blender")
    return mesh


if __name__ == "__main__":
    # cfg = WallMeshPartsCfg(wall_edges=("left", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    # # 
    # cfg = WallMeshPartsCfg(wall_edges=("up", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    # 
    # cfg = WallMeshPartsCfg(wall_edges=("right", "bottom"))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()

    cfg = WallMeshPartsCfg(wall_edges=("middle_right", "middle_bottom"))
    mesh = create_wall_mesh(cfg)
    mesh.show()

    for i in range(10):
        cfg = WallMeshPartsCfg(wall_edges=("middle_right", "middle_left"), door_direction="up")
        mesh = create_wall_mesh(cfg)
        mesh.show()
