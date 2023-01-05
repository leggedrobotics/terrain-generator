import time
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mesh_parts.mesh_utils import visualize_mesh

import argparse


def visualize_meshes(meshes):
    # Visualize meshes one by one with Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(meshes[0])
    for i, mesh in enumerate(meshes):
        print("mesh ", mesh)
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])
        # geometry = mesh

        R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
        print("R ", R)
        mesh.rotate(R, center=[0, 0, 0])

        vis.clear_geometries()

        # ctr = vis.get_view_control()
        # ctr.set_front([0, 1, 0])
        # ctr.set_up([0, 0, 1])
        # ctr.change_field_of_view(step=30)

        # ctr.camera_local_rotate(0, 0.5, 0)
        # ctr.set_zoom(3.5)

        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        # Add some time delay to see the mesh
        time.sleep(1)

        # image = vis.capture_screen_float_buffer(False)
        # # plt.imshow(np.asarray(image))
        # image = np.asarray(image)
        # print("images ", image)
        # # image = np.flip(image, 2)
        # # Save the image using OpenCV
        # cv2.imwrite(f"image_{i}.jpg", image)
        vis.capture_screen_image(f"image_{i}.jpg")
        # plt.imsave(np.asarray(image), f"test_{i}.png")


def visualize_and_save_mesh(mesh, vis=None, save_path=None):
    # Visualize meshes one by one with Open3D
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    mesh.compute_vertex_normals()
    R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
    mesh.rotate(R, center=[0, 0, 0])

    vis.clear_geometries()

    # vis.add_geometry(mesh)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    # Add some time delay to see the mesh
    time.sleep(1)

    vis.capture_screen_image(save_path)
    return vis
    # vis.destroy_window()


def main():
    # Load meshes
    meshes = []
    vis = None
    for i in range(0, 10):
        mesh = o3d.io.read_triangle_mesh(f"results/result_mesh_{i}.stl")
        # mesh = trimesh.load_mesh(f"results/result_mesh_{i}.stl")
        vis = visualize_and_save_mesh(mesh, vis, save_path=f"results/result_mesh_{i}.png")

    vis.destroy_window()
    # meshes.append(mesh)

    # Visualize meshes one by one with Open3D
    # visualize_meshes(meshes)


def show(mesh_path):
    # Load meshes
    mesh = trimesh.load_mesh(mesh_path)
    visualize_mesh(mesh)


if __name__ == "__main__":
    # main()
    # show()

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default="results/result_mesh_0.stl")
    args = parser.parse_args()

    show(args.mesh_path)
