#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
import argparse
import numpy as np
import trimesh
from typing import Optional
import random

from terrain_generator.wfc.wfc import WFCSolver

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_pattern, get_mesh_gen

# from trimesh_tiles.mesh_parts.overhanging_parts import FloorOverhangingParts
from terrain_generator.utils.mesh_utils import visualize_mesh, compute_sdf
from terrain_generator.utils import calc_spawnable_locations_with_sdf
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    OverhangingMeshPartsCfg,
    OverhangingBoxesPartsCfg,
)
from terrain_generator.trimesh_tiles.mesh_parts.overhanging_parts import get_cfg_gen

from configs.navigation_cfg import IndoorNavigationPatternLevels
from configs.overhanging_cfg import OverhangingTerrainPattern, OverhangingPattern, OverhangingFloorPattern
from terrain_generator.navigation.mesh_terrain import MeshTerrain, MeshTerrainCfg
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
    overhanging_cfg: Optional[OverhangingPattern] = None,
    prefix="mesh",
    mesh_dir="results/result",
    shape=[20, 20],
    initial_tile_name="floor",
    overhanging_initial_tile_name="walls_empty",
    visualize=False,
    enable_history=False,
    enable_sdf=False,
    sdf_resolution=0.1,
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
    save_dir = os.path.join(mesh_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)

    tiles, wave, wave_order, wave_names = solve_with_wfc(cfg, shape, initial_tile_name)

    if overhanging_cfg is not None:
        over_tiles, over_wave, over_wave_order, over_wave_names = solve_with_wfc(
            overhanging_cfg, shape, overhanging_initial_tile_name
        )

    # Save the history of the solver
    if enable_history:
        history_dir = os.path.join(mesh_dir, f"{prefix}_history")
        os.makedirs(history_dir, exist_ok=True)
        np.save(os.path.join(history_dir, "wave.npy"), wave)
        np.save(os.path.join(history_dir, "wave_order.npy"), wave_order)

        # We can visualize the wave propagation. We save the mesh for each step.
        parts_dir = os.path.join(history_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        translated_parts_dir = os.path.join(history_dir, "translated_parts")
        os.makedirs(translated_parts_dir, exist_ok=True)

    print("Converting to mesh...")
    # Compose the whole mesh from the tiles
    result_mesh = trimesh.Trimesh()
    if overhanging_cfg is not None:
        result_terrain_mesh = trimesh.Trimesh()
        result_overhanging_mesh = trimesh.Trimesh()

    if enable_sdf:
        sdf_dim = np.array(cfg.dim) * 3  # to merge with neighboring tiles
        sdf_dim[2] = 3.0 * 2
        sdf_array_dim = (np.array(wave.shape) + 2) * cfg.dim[:2] / sdf_resolution
        sdf_array_dim = np.array([sdf_array_dim[0], sdf_array_dim[1], sdf_dim[2] / sdf_resolution], dtype=int)
        sdf_min = np.inf * np.ones(sdf_array_dim, dtype=np.float32)

    with alive_bar(len(wave.flatten())) as bar:
        for y in range(wave.shape[0]):
            for x in range(wave.shape[1]):
                terrain_mesh = tiles[wave_names[wave[y, x]]].get_mesh().copy()
                mesh = terrain_mesh.copy()

                if overhanging_cfg is not None:
                    over_mesh = trimesh.Trimesh()
                    # terrain_mesh += mesh
                    if np.random.rand() < overhanging_cfg.overhanging_prob:
                        mesh_cfg = random.choice(overhanging_cfg.overhanging_cfg_list)
                        mesh_cfg.mesh = mesh
                        # over_box_mesh_cfg = create_overhanging_boxes(mesh_cfg)
                        cfg_gen = get_cfg_gen(mesh_cfg)
                        over_mesh_cfg = cfg_gen(mesh_cfg)
                        over_box_mesh = get_mesh_gen(over_mesh_cfg)(over_mesh_cfg)
                        over_mesh += over_box_mesh
                    # over_mesh += over_tiles[over_wave_names[over_wave[y, x]]].get_mesh().copy()
                    mesh += over_mesh
                if enable_sdf:
                    # Compute SDF around the mesh
                    mesh_sdf = compute_sdf(mesh, dim=sdf_dim, resolution=sdf_resolution)
                    x_min = int(x * cfg.dim[0] / sdf_resolution)
                    y_min = int((wave.shape[0] - y - 1) * cfg.dim[1] / sdf_resolution)
                    x_max = int((x + 2 + 1) * cfg.dim[0] / sdf_resolution)
                    y_max = int((wave.shape[0] - y + 2) * cfg.dim[1] / sdf_resolution)
                    # Update sdf_min by comparing the relevant part
                    sdf_min[x_min:x_max, y_min:y_max, :] = np.minimum(sdf_min[x_min:x_max, y_min:y_max, :], mesh_sdf)

                # save original parts for visualization
                if enable_history:
                    mesh.export(os.path.join(parts_dir, f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}.obj"))
                    if overhanging_cfg is not None:
                        over_mesh.export(
                            os.path.join(parts_dir, f"{over_wave[y, x]}_{y}_{x}_{over_wave_names[over_wave[y, x]]}.obj")
                        )
                        terrain_mesh.export(
                            os.path.join(parts_dir, f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}_terrain.obj")
                        )
                # Translate to the position of the tile
                xy_offset = np.array([x * cfg.dim[0], -y * cfg.dim[1], 0.0])
                mesh.apply_translation(xy_offset)
                if overhanging_cfg is not None:
                    over_mesh.apply_translation(xy_offset)
                    terrain_mesh.apply_translation(xy_offset)
                if enable_history:
                    mesh.export(
                        os.path.join(
                            translated_parts_dir, f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}_translated.obj"
                        )
                    )
                    if overhanging_cfg is not None:
                        over_mesh.export(
                            os.path.join(
                                translated_parts_dir,
                                f"{over_wave[y, x]}_{y}_{x}_{over_wave_names[over_wave[y, x]]}_translated.obj",
                            )
                        )
                        terrain_mesh.export(
                            os.path.join(
                                translated_parts_dir,
                                f"{wave[y, x]}_{y}_{x}_{wave_names[wave[y, x]]}_terrain_translated.obj",
                            )
                        )
                result_mesh += mesh
                if overhanging_cfg is not None:
                    result_terrain_mesh += terrain_mesh
                    result_overhanging_mesh += over_mesh
                bar()

    bbox = result_mesh.bounding_box.bounds
    print("bbox = ", bbox)
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Translate the mesh to the center of the bounding box.
    result_mesh = result_mesh.apply_translation(-center)

    print("saving mesh to ", save_dir)
    result_mesh.export(os.path.join(save_dir, "mesh.obj"))
    # result_mesh.export(save_dir)
    if overhanging_cfg is not None:
        result_terrain_mesh = result_terrain_mesh.apply_translation(-center)
        result_terrain_mesh.export(os.path.join(save_dir, "terrain_mesh.obj"))
        # result_terrain_mesh.export(save_dir + "_terrain.obj")
        result_overhanging_mesh = result_overhanging_mesh.apply_translation(-center)
        result_overhanging_mesh.export(os.path.join(save_dir, "overhanging_mesh.obj"))
        # result_overhanging_mesh.export(save_dir + "_overhanging.obj")

    if enable_sdf:
        # sdf_name = save_dir + ".npy"
        sdf_name = os.path.join(save_dir, "sdf.npy")
        print("saving sdf to ", sdf_name)
        sdf_tile_n = (np.array(cfg.dim[:2]) / sdf_resolution).astype(int)
        sdf = sdf_min[sdf_tile_n[1] : -sdf_tile_n[1], sdf_tile_n[0] : -sdf_tile_n[0], :]
        np.save(sdf_name, sdf)
        if overhanging_cfg is None:
            result_terrain_mesh = result_mesh
        spawnable_locations = calc_spawnable_locations_with_sdf(
            result_terrain_mesh, sdf_min, height_offset=0.5, sdf_resolution=sdf_resolution, sdf_threshold=0.4
        )
        # locations_name = save_dir + "spawnable_locations.npy"
        locations_name = os.path.join(save_dir, "spawnable_locations.npy")
        print("saving locations to ", locations_name)
        np.save(locations_name, spawnable_locations)

        # Save as mesh_terrain
        mesh_cfg = MeshTerrainCfg(mesh=result_terrain_mesh, sdf=sdf, sdf_resolution=sdf_resolution)
        mesh_terrain = MeshTerrain(mesh_cfg)
        mesh_terrain.save(save_dir)
    if visualize:
        visualize_mesh(result_mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mesh from configuration")
    parser.add_argument(
        "--cfg",
        type=str,
        choices=["indoor", "overhanging", "overhanging_floor"],
        default="indoor",
        help="Which configuration to use",
    )
    parser.add_argument("--over_cfg", action="store_true", help="Whether to use overhanging configuration")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the generated mesh")
    parser.add_argument("--enable_history", action="store_true", help="Whether to enable mesh history")
    parser.add_argument("--enable_sdf", action="store_true", help="Whether to enable sdf")
    parser.add_argument("--initial_tile_name", type=str, default="floor", help="Whether to enable sdf")
    parser.add_argument(
        "--mesh_dir", type=str, default="results/generated_terrain", help="Directory to save the generated mesh files"
    )
    parser.add_argument("--mesh_name", type=str, default="mesh", help="Base name of the generated mesh files")
    parser.add_argument("--num_terrains", type=int, default=1, help="Number of terrains to generate")
    parser.add_argument("--sdf_resolution", type=float, default=0.1, help="Resolution for generating SDF")
    args = parser.parse_args()

    if args.cfg == "indoor":
        cfg = IndoorNavigationPatternLevels(wall_height=3.0)
    elif args.cfg == "overhanging":
        cfg = OverhangingTerrainPattern()
    elif args.cfg == "overhanging_floor":
        cfg = OverhangingFloorPattern()
    else:
        raise ValueError(f"Unknown configuration: {args.cfg}")

    if args.over_cfg:
        over_cfg = OverhangingPattern()
    else:
        over_cfg = None

    for i in range(args.num_terrains):
        mesh_prefix = f"{args.mesh_name}_{i}"
        create_mesh_from_cfg(
            cfg,
            over_cfg,
            prefix=mesh_prefix,
            initial_tile_name=args.initial_tile_name,
            mesh_dir=args.mesh_dir,
            visualize=args.visualize,
            enable_history=args.enable_history,
            enable_sdf=args.enable_sdf,
            sdf_resolution=args.sdf_resolution
        )
