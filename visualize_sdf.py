import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Load SDF array
sdf_array = np.load("results/overhanging_with_sdf/mesh_0.obj.npy")
mesh = trimesh.load("results/overhanging_with_sdf/mesh_0.obj")

# Define voxel size (the spacing between SDF grid points)
voxel_size = 0.10

# Compute SDF grid origin (the position of the first SDF grid point)
sdf_origin = -np.array([sdf_array.shape[0], -sdf_array.shape[1], sdf_array.shape[2]]) * voxel_size / 2

# Create meshgrid of SDF grid points
grid_x, grid_y, grid_z = np.meshgrid(
    np.arange(sdf_array.shape[0]), np.arange(sdf_array.shape[1]), np.arange(sdf_array.shape[2]), indexing="ij"
)
sdf_points = np.stack(
    [
        grid_y.flatten() * voxel_size + sdf_origin[0],
        -grid_x.flatten() * voxel_size + sdf_origin[1],
        grid_z.flatten() * voxel_size + sdf_origin[2],
    ],
    axis=1,
)
print("sdf_points ", sdf_points.shape)

# # Threshold SDF values to create a binary occupancy grid
threshold = 0.10
occupancy = (sdf_array > threshold).flatten()
print("occupancy ", occupancy.shape)
#
# # Extract the occupied SDF grid points and their SDF values
points = sdf_points[occupancy]
sdf_values = sdf_array[occupancy.reshape(sdf_array.shape)]
# points = sdf_points
# sdf_values = sdf_array

# Create a point cloud where the points are the occupied SDF grid points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Set the point cloud colors based on the SDF values
cmap = plt.get_cmap("rainbow")
norm = plt.Normalize(vmin=0.0, vmax=2.0)
sdf_colors = cmap(norm(sdf_values.flatten()))[:, :3]
print("sdf_colors ", sdf_colors.shape)
pcd.colors = o3d.utility.Vector3dVector(sdf_colors)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
# o3d.visualization.draw_geometries([voxel_grid])

# Visualize the point cloud and the mesh
o3d_mesh = mesh.as_open3d
o3d_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([voxel_grid, o3d_mesh])
# o3d.visualization.draw_geometries([pcd, o3d_mesh])
# o3d.visualization.draw_geometries([pcd])
