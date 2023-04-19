import numpy as np
import torch
import trimesh

from ..utils import (
    calc_spawnable_locations_on_terrain,
    locations_to_graph,
    visualize_mesh_and_graphs,
    filter_spawnable_locations_with_sdf,
    get_height_array_of_mesh_with_resolution,
    distance_matrix_from_graph,
    create_2d_graph_from_height_array,
    visualize_distance,
)

from ..navigation.mesh_terrain import MeshTerrain, MeshTerrainCfg, SDFArray, NavDistance


def test_mesh_terrain_transform(visualize):
    nav_mesh = MeshTerrain(MeshTerrainCfg())
    translation = torch.Tensor([1.0, 2.0, 3.0])

    # translation
    nav_mesh.translate(translation)
    assert np.allclose(nav_mesh.sdf.center.cpu().numpy(), translation, atol=1e-5)
    assert np.allclose(nav_mesh.nav_distance.center.cpu().numpy(), translation[:2], rtol=1e-3, atol=1e-5)

    # transform
    transform = np.eye(4)
    transform[:3, 3] = translation
    nav_mesh.transform(transform)
    assert np.allclose(nav_mesh.sdf.center, translation * 2, atol=1e-5)
    assert np.allclose(nav_mesh.nav_distance.center, translation[:2] * 2, rtol=1e-3, atol=1e-5)
