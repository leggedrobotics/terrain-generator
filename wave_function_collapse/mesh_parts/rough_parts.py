import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mesh_parts.mesh_parts_cfg import (
    WallMeshPartsCfg,
    # StairMeshPartsCfg,
    HeightMapMeshPartsCfg,
)
from mesh_parts.mesh_utils import (
    merge_meshes,
    rotate_mesh,
    ENGINE,
    get_height_array_of_mesh,
)
from mesh_parts.basic_parts import create_floor
from mesh_parts.create_tiles import create_mesh_tile

from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d


def generate_perlin_tile_configs(name, dim, weight, height=0.5, offset=0.1, seed=0, shape=(128, 128), res=(8, 8)):

    np.random.seed(seed)
    noise = generate_perlin_noise_2d(shape, res, tileable=(True, True))
    xx = np.linspace(-1, 1, shape[0])
    yy = np.linspace(-1, 1, shape[1])
    x, y = np.meshgrid(xx, yy)
    base_edge_array = np.clip(1.0 - (x**8 + y**8), 0.0, 1.0)
    noise *= base_edge_array
    cfgs = []

    def generate_cfgs(noise, name):
        noise = noise * 0.1 + 0.1

        # edge_array = np.zeros(shape)
        # xx = np.linspace(-1, 1, shape[0])
        # yy = np.linspace(-1, 1, shape[1])
        # x, y = np.meshgrid(xx, yy)

        step_height = 1.0 / (shape[1] - 2)
        arrays = []
        ramp_patterns = ["flat", "mountain", "wide", "corner", "corner_flipped"]
        for ramp_pattern in ramp_patterns:
            base_edge_array = np.clip(1.0 - (x**8 + y**8), 0.0, 1.0)

            if ramp_pattern == "flat":
                # edge_array = base_edge_array
                offset_array = np.zeros(shape)
                # if offset > 0.1:
                edge_array = np.ones(shape)

                new_array = noise * edge_array + offset + offset_array
                arrays.append(new_array)
            elif ramp_pattern == "mountain":
                edge_array = np.clip(1.0 - (x**2 + y**2), 0.0, 1.0)
                offset_array = edge_array * height
                if offset > 0.1:
                    edge_array = np.ones(shape)

                new_array = noise * edge_array + offset + offset_array
                arrays.append(new_array)
            else:
                array_1 = np.zeros(shape)
                array_2 = np.zeros(shape)
                h_1 = 0.0
                h_2 = 1.0
                for s in range(shape[0]):
                    if ramp_pattern == "wide":
                        array_1[:, s] = h_1
                        array_2[:, s] = h_2
                    elif ramp_pattern == "corner":
                        array_1[:s, s] = h_1
                        array_2[:s, s] = h_2
                        array_1[s, :s] = h_1
                        array_2[s, :s] = h_2
                        array_1[s, s] = h_1
                        array_2[s, s] = h_2
                    elif ramp_pattern == "corner_flipped":
                        array_1[:s, s] = 1.0 - h_1
                        array_2[:s, s] = 1.0 - h_2
                        array_1[s, :s] = 1.0 - h_1
                        array_2[s, :s] = 1.0 - h_2
                        array_1[s, s] = 1.0 - h_1
                        array_2[s, s] = 1.0 - h_2
                    if s > 0:
                        h_1 = min(h_1 + step_height, 1.0)
                        h_2 = max(h_2 - step_height, 0.0)
                # edge_array = array_1

                if offset > 0.1:
                    array_1 = np.ones(shape)
                    array_2 = np.ones(shape)

                offset_array = array_1 * height
                new_array = noise * array_1 + offset + offset_array
                arrays.append(new_array)

                offset_array = array_2 * height
                new_array = noise * array_2 + offset + offset_array
                arrays.append(new_array)

        weight_per_tile = weight / (len(ramp_patterns))
        # print("weight_per_tile", weight_per_tile)
        cfgs = []
        for i, array in enumerate(arrays):
            cfg = HeightMapMeshPartsCfg(
                name=f"{name}_{i}",
                dim=dim,
                height_map=array,
                rotations=(90, 180, 270),
                flips=("x", "y"),
                weight=weight_per_tile,
                slope_threshold=0.5,
                target_num_faces=1000,
                simplify=False,
            )
            cfgs.append(cfg)
        return cfgs

    cfgs += generate_cfgs(noise, name)
    noise = np.rot90(noise, 1)
    cfgs += generate_cfgs(noise, name + "_rotated")
    return cfgs


if __name__ == "__main__":
    print("test")
    # generate_perlin_tile_configs("perlin", 1.0)
    cfgs = []
    cfgs += generate_perlin_tile_configs("perlin", [2, 2, 2], weight=1.0)
    cfgs += generate_perlin_tile_configs("perlin_0.5", [2, 2, 2], weight=1.0, offset=0.5, height=1.0)
    print(cfgs)
    for cfg in cfgs:
        tile = create_mesh_tile(cfg)
        mesh = tile.get_mesh()
        print(get_height_array_of_mesh(mesh, cfg.dim, 5))
        mesh.show()
