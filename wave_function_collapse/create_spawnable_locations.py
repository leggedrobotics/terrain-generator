import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from mesh_parts.mesh_utils import get_height_array_of_mesh


def create_spawnable_locations(mesh, array_shape):
    """
    Create spawnable locations from a mesh.
    :param mesh: Mesh to create spawnable locations from.
    :param array_shape: Shape of the array to create spawnable locations from.
    :return: Spawnable locations.
    """
    # Get the bounding box of the mesh.
    bbox = mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Get the size of the bounding box.
    mesh = mesh.apply_translation(-center)

    bbox = mesh.bounding_box.bounds
    size = np.max(bbox, axis=0) - np.min(bbox, axis=0)
    # Get the minimum and maximum of the bounding box.
    b_min = np.min(bbox, axis=0)
    b_max = np.max(bbox, axis=0)
    dim = np.array([b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2]])
    n_points = int((b_max[0] - b_min[0]) * 5)
    array = get_height_array_of_mesh(mesh, dim, n_points)

    # plt.imshow(array)
    # plt.colorbar()
    # plt.show()

    flat_filter = np.ones((7, 7)) * -1
    flat_filter[2, 2] = flat_filter.shape[0] * flat_filter.shape[1] - 1

    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_img = cv2.filter2D(array, -1, flat_filter)
    # plt.imshow(filtered_img)
    # plt.colorbar()
    # plt.show()

    # print("filetered_img", filtered_img)
    # plt.imshow(np.abs(filtered_img) <= 0.01)
    # plt.colorbar()
    # plt.show()

    spawnable_img = np.argwhere(np.abs(filtered_img) <= 0.01)
    # print("spawnable_locations", spawnable_locations)
    # # make it sparse
    # spawnable_locations = spawnable_locations[::10]
    # print("spawnable_locations", spawnable_locations)
    spawnable_idx = np.random.choice(np.arange(len(spawnable_img)), 1000)
    spawnable_locations = spawnable_img[spawnable_idx]

    spawnable_locations = spawnable_locations * (dim[:2] / n_points) + b_min[:2]
    # swap 0 and 1
    spawnable_locations[:, [0, 1]] = spawnable_locations[:, [1, 0]]
    spawnable_locations[:, 1] *= -1
    print("spawnable_locations", spawnable_locations)

    return mesh, spawnable_img, spawnable_locations


def visualize_spawnable_locations(mesh, spawnable_locations):
    if isinstance(mesh, trimesh.Trimesh):
        o3d_mesh = mesh.as_open3d
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d_mesh = mesh
    # Visualize meshes one by one with Open3D
    o3d_mesh.compute_vertex_normals()
    # R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
    # o3d_mesh.rotate(R, center=[0, 0, 0])
    # o3d.visualization.draw_geometries([o3d_mesh])
    shape = spawnable_locations.shape
    points_below = np.zeros((spawnable_locations.shape[0], 3))
    points_below[:, :2] = spawnable_locations
    points_below[:, 2] = -1.0

    points_above = np.zeros((spawnable_locations.shape[0], 3))
    points_above[:, :2] = spawnable_locations
    points_above[:, 2] = 2.0
    points = np.concatenate([points_below, points_above], axis=0)
    lines = np.arange(spawnable_locations.shape[0])
    lines = np.stack([lines, lines + spawnable_locations.shape[0]], axis=1).reshape(-1, 2)
    print("points ", points)
    print("lines ", lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(o3d_mesh)
    vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([o3d_mesh, line_set], zoom=0.8)


# vis = o3d.visualization.Visualizer()
# vis.capture_screen_image(save_path)


if __name__ == "__main__":
    # Load the mesh.
    mesh = trimesh.load("results/results_5/result_mesh_0.stl")
    # Create spawnable locations.
    mesh, spawnable_img, spawnable_locations = create_spawnable_locations(mesh, [500, 500])
    # save
    np.save("spawnable_locations.npy", spawnable_locations)
    np.save("spawnable_img.npy", spawnable_img)
    mesh.export("mesh_0.obj")

    spawnable_locations = np.load("spawnable_locations.npy")
    mesh = trimesh.load("mesh_0.obj")
    visualize_spawnable_locations(mesh, spawnable_locations)
