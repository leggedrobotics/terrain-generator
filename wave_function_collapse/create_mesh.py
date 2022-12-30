import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MeshDef:
    name: str
    file: str
    scale: float = 1.0
    offset: np.ndarray = np.zeros(3)
    rotation: float = 0.0
    color: np.ndarray = np.array([0.5, 0.5, 0.5, 1.0])


@dataclass
class Meshes:
    meshes: Tuple[MeshDef, ...] = (
        # MeshDef("F", "", 1.0, np.array([0.0, 0.0, 0.0])),
        MeshDef("FA", "floor_01.stl", 1.0, np.array([0.0, 0.0, 0.0]), 0.0),
        MeshDef("FB", "wall_02.stl", 1.0, np.array([0.0, 0.0, -1.5]), 0.0),
        MeshDef("FC", "wall_06.stl", 1.0, np.array([0.0, 0.0, -1.5]), 0.0),
        MeshDef("WA", "wall_05.stl", 1.0, np.array([0.0, 0.0, 0.0]), 0.0),
        MeshDef("WC", "wall_02.stl", 1.0, np.array([0.0, 0.0, 0.0]), 0.0),
        MeshDef("WI", "wall_06.stl", 1.0, np.array([0.0, 0.0, 0.0]), 180.0),
        MeshDef("WH", "wall_01.stl", 1.0, np.array([0.0, 0.0, 0.0]), 90.0),
    )
    root_dir: str = "../parts/walls/"

    def get_mesh(self, name: str) -> trimesh.Trimesh:
        for mesh in self.meshes:
            if mesh.name in name:
                print("mesh file ", mesh.file)
                filename = self.root_dir + mesh.file
                if len(mesh.file) > 0:
                    stl = trimesh.load_mesh(filename, process=False)

                    transform = trimesh.transformations.rotation_matrix(mesh.rotation * np.pi / 180, [0, 0, 1])
                    stl.apply_transform(transform)

                    if mesh.name == name:
                        pass
                    elif "_x" in name:
                        # Create the transformation matrix for inverting the mesh in the x-axis
                        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [1, 0, 0])
                        # Apply the transformation to the mesh
                        stl.apply_transform(transform)
                    elif "_y" in name:
                        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 1, 0])
                        # Apply the transformation to the mesh
                        stl.apply_transform(transform)
                    if "90" in name:
                        # Create the transformation matrix for rotating the mesh around the z-axis by 90 degrees
                        angle = -np.pi / 2  # 90 degrees in radians
                        axis = [0, 0, 1]  # The z-axis
                        transform = trimesh.transformations.rotation_matrix(angle, axis)
                        # Apply the transformation to the mesh
                        stl.apply_transform(transform)
                        # stl = stl.rotate([0, 0, 1], np.pi / 2)
                    elif "180" in name:
                        angle = -np.pi  # 180 degrees in radians
                        axis = [0, 0, 1]  # The z-axis
                        transform = trimesh.transformations.rotation_matrix(angle, axis)
                        # Apply the transformation to the mesh
                        stl.apply_transform(transform)
                        # stl = stl.rotate([0, 0, 1], np.pi)
                    elif "270" in name:
                        angle = -1.5 * np.pi  # 270 degrees in radians
                        axis = [0, 0, 1]  # The z-axis
                        transform = trimesh.transformations.rotation_matrix(angle, axis)
                        # Apply the transformation to the mesh
                        stl.apply_transform(transform)
                        # stl = stl.rotate([0, 0, 1], np.pi * 1.5)
                    stl.apply_scale(mesh.scale)
                    stl.apply_translation(mesh.offset)
                    return stl
                else:
                    return trimesh.Trimesh()


def create_mesh(filename, names):
    wave = np.load(filename)
    names = np.load(names)
    meshes = Meshes()
    print("wave ", wave)
    print("names ", names)

    result_mesh = trimesh.Trimesh()
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            print("name ", names[wave[y, x]])
            mesh = meshes.get_mesh(names[wave[y, x]])
            xy_offset = np.array([x, y, 0.0])
            mesh.apply_translation(xy_offset)
            result_mesh += mesh

    result_mesh.export("result_mesh.stl")
    result_mesh.show()
    # mesh.export(f"meshes/{x}_{y}_{z}.stl")


create_mesh("wave.npy", "names.npy")
