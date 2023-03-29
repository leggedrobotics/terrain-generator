import os
import numpy as np
import trimesh
from typing import Optional

from wfc.wfc import WFCSolver

from trimesh_tiles.mesh_parts.create_tiles import create_mesh_pattern
from utils.mesh_utils import visualize_mesh
from trimesh_tiles.mesh_parts.mesh_parts_cfg import MeshPattern, OverhangingMeshPartsCfg

from configs.navigation_cfg import IndoorNavigationPatternLevels
from configs.overhanging_cfg import OverhangingTerrainPattern, OverhangingPattern
from alive_progress import alive_bar


def solve_with_wfc(cfg: MeshPattern, shape, initial_tile_name):
    # Create tiles from config
    print("Creating tiles...")
    tiles = create_mesh_pattern(cfg)

    # Create solver
    wfc_solver = WFCSolver(shape=shape, dimensions=2, seed=None)

    # Add tiles to the solver
    for tile in tiles.values():
        wfc_solver.register_tile(*tile.get_dict_tile())
        # print(f"Tile: {tile.name}, {tile.array}")

    # Place initial tile in the center.
    init_tiles = [(initial_tile_name, (wfc_solver.shape[0] // 2, wfc_solver.shape[1] // 2))]
    wave = wfc_solver.run(init_tiles=init_tiles, max_steps=10000)
    wave_order = wfc_solver.wfc.wave.wave_order
    names = wfc_solver.names
    return tiles, wave, wave_order, names


def create_mesh_from_cfg(
    cfg: MeshPattern,
    overhanging_cfg: Optional[MeshPattern] = None,
    mesh_name="result_mesh.obj",
    mesh_dir="results/result",
    shape=[20, 20],
    initial_tile_name="floor",
    overhanging_initial_tile_name="walls_empty",
    visualize=False,
    enable_history=False,
):
    """Generate terrain mesh from config.
    It will generate a mesh from the given config and save it to the given path.
    Args:
        cfg: MeshPattern config
        mesh_name: name of the mesh file
        mesh_dir: directory to save the mesh file
        shape: shape of the whole tile.
        initial_tile_name: name of the initial tile which is positioned at the center of the tile.
        visualize: visualize the mesh or not
        enable_history: save the history of the solver or not
    """
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_name = os.path.join(mesh_dir, mesh_name)

    tiles, wave, wave_order, wave_names = solve_with_wfc(cfg, shape, initial_tile_name)

    if overhanging_cfg is not None:
        print("overhanging_cfg ", overhanging_cfg)
        over_tiles, over_wave, over_wave_order, over_wave_names = solve_with_wfc(
            overhanging_cfg, shape, overhanging_initial_tile_name
        )
        print("over_wave ", over_wave)

    # Save the history of the solver
    if enable_history:
        np.save(os.path.join(mesh_dir, "wave.npy"), wave)
        np.save(os.path.join(mesh_dir, "wave_order.npy"), wave_order)

    print("Converting to mesh...")
    # If history is enabled, we can visualize the wave propagation. We save the mesh for each step.
    if enable_history:
        parts_dir = os.path.join(mesh_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        translated_parts_dir = os.path.join(mesh_dir, "translated_parts")
        os.makedirs(translated_parts_dir, exist_ok=True)

    # Compose the whole mesh from the tiles
    result_mesh = trimesh.Trimesh()
    with alive_bar(len(wave.flatten())) as bar:
        for y in range(wave.shape[0]):
            for x in range(wave.shape[1]):
                mesh = tiles[wave_names[wave[y, x]]].get_mesh().copy()

                if overhanging_cfg is not None:
                    over_mesh = over_tiles[over_wave_names[over_wave[y, x]]].get_mesh().copy()
                    mesh += over_mesh

                # save original parts for visualization
                if enable_history:
                    mesh.export(os.path.join(parts_dir, f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}.obj"))
                # Translate to the position of the tile
                xy_offset = np.array([x * cfg.dim[0], -y * cfg.dim[1], 0.0])
                mesh.apply_translation(xy_offset)
                if enable_history:
                    mesh.export(
                        os.path.join(
                            translated_parts_dir, f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}_translated.obj"
                        )
                    )
                result_mesh += mesh
                bar()

    bbox = result_mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Translate the mesh to the center of the bounding box.
    result_mesh = result_mesh.apply_translation(-center)

    print("saving mesh to ", mesh_name)
    result_mesh.export(mesh_name)
    if visualize:
        visualize_mesh(result_mesh)


if __name__ == "__main__":
    # cfg = IndoorNavigationPatternLevels(wall_height=3.0)
    for i in range(10):
        cfg = OverhangingTerrainPattern()
        over_cfg = OverhangingPattern()
        create_mesh_from_cfg(
            cfg,
            over_cfg,
            mesh_name=f"test_mesh_{i}.obj",
            mesh_dir="results/test_overhanging",
            visualize=False,
            enable_history=False,
        )
