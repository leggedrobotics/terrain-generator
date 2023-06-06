import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from terrain_generator.utils import visualize_mesh_and_sdf_values
from terrain_generator.navigation.mesh_terrain import MeshTerrain, MeshTerrainCfg, SDFArray, NavDistance

# Load SDF array
sdf_array = np.load("results/overhanging_with_sdf_no_wall/mesh_1.obj.npy")
mesh = trimesh.load("results/overhanging_with_sdf_no_wall/mesh_1.obj")

sdf = SDFArray(sdf_array, resolution=0.1)

bbox = mesh.bounds
b_min = np.min(bbox, axis=0)
b_min[1] += 12.0
# b_max = np.max(bbox, axis=0)
b_max = b_min + np.array([30, 0.2, 3.0])

print("bbox ", bbox, "b_min", b_min, "b_max", b_max)
# dim = np.array(dim)
# num_elements = np.ceil(np.array(dim) / resolution).astype(int)
xyz_range = [np.linspace(b_min[i], b_max[i], num=int((b_max[i] - b_min[i]) * 50)) for i in range(3)]
query_points = np.stack(np.meshgrid(*xyz_range), axis=-1).astype(np.float32)
query_points = query_points.reshape(-1, 3)
print("query_points", query_points.shape)

sdf_values = sdf.get_sdf(query_points)
print("sdf_values", sdf_values.shape)

visualize_mesh_and_sdf_values(mesh, query_points, sdf_values)


# visualize_mesh_and_sdf(mesh, sdf_array, 0.1, vmin=0.0, vmax=3.0)
