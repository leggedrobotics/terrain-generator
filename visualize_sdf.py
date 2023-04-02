import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from utils import visualize_mesh_and_sdf

# Load SDF array
sdf_array = np.load("results/overhanging_with_sdf/mesh_0.obj.npy")
mesh = trimesh.load("results/overhanging_with_sdf/mesh_0.obj")


visualize_mesh_and_sdf(mesh, sdf_array, 0.1, vmin=0.0, vmax=3.0)
