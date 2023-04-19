import trimesh
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

# Set parameters
size = 100  # size of the mesh (increased for higher resolution)
threshold = 0.45  # threshold for extracting the cave mesh (adjusted for denser mesh)
subdivisions = 3  # number of times to subdivide the mesh for smoothing

# Generate the noise function using pyvista
def generate_noise(amplitude, freq, offset, grid_size):
    noise = pv.perlin_noise(amplitude, freq, offset)
    grid = pv.sample_function(noise, [-3, 3.0, -3, 3.0, -1.0, 1.0], dim=(grid_size, grid_size, grid_size))
    return grid["scalars"].reshape(grid_size, grid_size, grid_size)


grid_numpy = np.zeros((size, size, size))
grid_numpy += generate_noise(0.5, (1, 1, 1), (0, 0, 0), size)
grid_numpy += generate_noise(0.3, (2, 2, 2), (0, 0, 0), size)
# grid_numpy += generate_noise(0.3, (

# import trimesh
# import numpy as np
# import pyvista as pv
# from skimage.measure import marching_cubes
#
# # Set parameters
# size = 30  # size of the mesh
# threshold = 0.10  # threshold for extracting the cave mesh
# subdivisions = 3  # number of times to subdivide the mesh for smoothing
#
# # Generate the noise function using pyvista
# freq = (1, 1, 1)
# noise = pv.perlin_noise(1, freq, (0, 0, 0))
# grid = pv.sample_function(noise, [-1.0, 1.0, -2.0, 2.0, -0.5, 0.5], dim=(size, size, size))
# # grid = grid.threshold(threshold, invert=False)
#
# grid_numpy = grid["scalars"].reshape(size, size, size)
#
# # Calculate the level for marching cubes based on the data range
# min_val, max_val = np.min(grid_numpy), np.max(grid_numpy)
# print("min_val", min_val)
# print("max_val", max_val)
# # level = min_val + threshold * (max_val - min_val)
# # level = min_val + 0.5 * (max_val - min_val)
level = 0.10
# print("level", level)
#
# Extract the mesh using marching cubes algorithm
vertices, faces, _, _ = marching_cubes(grid_numpy, level=level)

# # Extract the mesh using marching cubes algorithm
# vertices, faces, _, _ = marching_cubes(grid_numpy, level=threshold * size)

# Create the cave mesh and subdivide it for smoothing
cave_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

for i in range(subdivisions):
    cave_mesh = cave_mesh.subdivide()

cave_mesh.show()


# exit(0)

import trimesh
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes


# Set parameters
size = 30  # size of the mesh
threshold = 0.10  # threshold for extracting the cave mesh
subdivisions = 3  # number of times to subdivide the mesh for smoothing

# Generate the noise function using pyvista
freq = (1, 1, 1)
noise = pv.perlin_noise(1, freq, (0, 0, 0))
grid = pv.sample_function(noise, [-3, 3.0, -3, 3.0, -1.0, 1.0], dim=(size, size, size))

freq = (2, 2, 2)
noise = pv.perlin_noise(1, freq, (0, 0, 0))
print("noise ", noise)
grid_noise = pv.sample_function(noise, [0, 3.0, -0, 3.0, 0, 1.0], dim=(size, size, size))
print("grid ", grid)
print("grid_noise ", grid_noise)

# grid = grid + grid_noise * 0.1

out = grid.threshold(threshold, invert=False)

mn, mx = [out["scalars"].min(), out["scalars"].max()]
clim = (mn, mx * 1.8)

# out.plot(
#     cmap="gist_earth_r",
#     background="white",
#     show_scalar_bar=False,
#     lighting=True,
#     clim=clim,
#     show_edges=False,
# )

# vtk_obj = pv.wrap(out)

# Extract the occupancy numpy array
# occupancy = np.array(vtk_obj.GetCellData().GetScalars())

# Get the dimensions of the grid
# dims = vtk_obj.GetDimensions()

# Reshape the occupancy numpy array to match the grid dimensions
# occupancy = occupancy.reshape(dims)
# print("Occupancy shape:", occupancy.shape)

# Convert the threshold data to a point cloud
point_cloud = out.extract_geometry()
# Convert the point cloud to a numpy array
points = np.array(point_cloud.points)

import open3d as o3d

pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
# voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1).create_dense(
#     width=3, height=3, depth=1, voxel_size=0.1, origin=np.array([0, 0, 0]), color=[1.0, 0.7, 0.0]
# )
print("voxels ", voxels)

# Convert the voxel grid to a numpy array
voxel_array = np.asarray(voxels.get_voxels(), dtype=bool).reshape(30, 30, 10)

# Print the shape of the voxel array
print(voxel_array.shape)
print("voxel_array ", voxel_array, voxel_array.sum())

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.estimate_normals()
# resolution = 0.05
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=resolution)
#
# # estimate radius for rolling ball
# # distances = pcd.compute_nearest_neighbor_distance()
# # avg_dist = np.mean(distances)
# # radius = 1.0 * avg_dist
# #
# # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
# #     pcd, o3d.utility.DoubleVector([radius, radius * 2])
# # )
#
# # create the triangular mesh with the vertices and faces from open3d
# cave_mesh = trimesh.Trimesh(
#     np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals)
# )


# print("points", points, points.shape)
#
# # voxels = trimesh.voxel.VoxelGrid(points, 0.10)
# voxel_pitch = 0.1
# voxel_radius = 0.1
# point = np.array([0, 0, 0])
# voxels = trimesh.voxel.creation.local_voxelize(points, point, voxel_pitch, voxel_radius, fill=True)
#
# # Convert the voxels to a mesh and subdivide it for smoothing
# mesh = voxels.as_boxes()
# cave_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

# vtk_obj = pv.wrap(threshold_data)
#
# # Extract the occupancy numpy array
# occupancy = np.array(vtk_obj.GetCellData().GetScalars())
#
# # Reshape the occupancy numpy array to match the grid dimensions
# dims = threshold_data.dimensions
# occupancy = occupancy.reshape(dims)
#
# # Print the shape of the occupancy numpy array
# print(occupancy.shape)

# print("threshold_data", threshold_data)
# threshold_data = threshold_data.cast_to_explicit_structured_grid()
# print("threshold_data", threshold_data)

# Extract the mesh using marching cubes algorithm
# marching_cubes_filter = pv.filters.MarchingCubes()
# marching_cubes_filter.set_contour_value(threshold)
# mesh = marching_cubes_filter.execute(threshold_data)
#
# # Create the cave mesh and subdivide it for smoothing
# cave_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces)

# out = threshold_data
#
# mn, mx = [out["scalars"].min(), out["scalars"].max()]
# clim = (mn, mx * 1.8)
#
# out.plot(
#     cmap="gist_earth_r",
#     background="white",
#     show_scalar_bar=False,
#     lighting=True,
#     clim=clim,
#     show_edges=False,
# )
# mesh = out.contour([1], method="marching_cubes")
# dist = np.linalg.norm(mesh.points, axis=1)
# mesh.plot(scalars=dist, smooth_shading=True, specular=5, cmap="plasma", show_scalar_bar=False)

# Create the cave mesh
def create_cave_mesh(threshold_data):
    # vertices, faces, _, _ = marching_cubes(threshold_data, level=0.5)
    # vertices, faces, _, _ = marching_cubes(threshold_data, pitch=1.0)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Extract the mesh using marching cubes algorithm
    # vertices, faces, _, _ = trimesh.voxel.marching_cubes(threshold_data, level=0.5, pitch=1.0)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_array, pitch=0.1)
    # mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=0.20)
    # mesh = voxel_volume.as_boxes().to_mesh()
    return mesh


#
#
# # Create the cave mesh and subdivide it for smoothing
cave_mesh = create_cave_mesh(out)
for i in range(subdivisions):
    cave_mesh = cave_mesh.subdivide()

cave_mesh.show()

# Export the mesh
trimesh.exchange.export.export_mesh(cave_mesh, "cave.stl", file_type="stl")
