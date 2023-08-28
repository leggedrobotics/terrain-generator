#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from typing import Callable, Any, Optional, Union, Tuple
from dataclasses import asdict, is_dataclass
from itertools import product
import copy
import json
from scipy.spatial.transform import Rotation


ENGINE = "blender"
# Cache dir is in the above directory as this file.
# CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__cache__/mesh_cache")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__cache__")
# print("CACHE_DIR", CACHE_DIR)
# CACHE_DIR =
# ENGINE = "scad"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, trimesh.Trimesh):
            return None
        return json.JSONEncoder.default(self, obj)


def cfg_to_hash(cfg, exclude_keys=["weight", "load_from_cache"]):
    """MD5 hash of a config."""
    import hashlib

    def tuple_to_str(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = tuple_to_str(v)
            if isinstance(k, tuple):
                new_d[str(k)] = v
            else:
                new_d[k] = v
        return new_d

    if isinstance(cfg, dict):
        cfg_dict = copy.deepcopy(cfg)
    elif is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        raise ValueError("cfg must be a dict or dataclass.")
    for key in exclude_keys:
        cfg_dict.pop(key, None)
    cfg_dict = tuple_to_str(cfg_dict)
    encoded = json.dumps(cfg_dict, sort_keys=True, cls=NpEncoder).encode()
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    # encoded = json.dumps(cfg, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_cached_mesh_gen(
    mesh_gen_fn: Callable[[Any], trimesh.Trimesh],
    cfg,
    verbose=False,
    use_cache=True,
) -> Callable[[], trimesh.Trimesh]:
    """Generate a mesh if there's no cache. If there's cache, load from cache."""
    code = cfg_to_hash(cfg)
    mesh_cache_dir = os.path.join(CACHE_DIR, "mesh_cache")
    os.makedirs(mesh_cache_dir, exist_ok=True)
    if hasattr(cfg, "name"):
        name = cfg.name
    else:
        name = ""

    mesh_name = f"{name}_{code}.obj"

    def mesh_gen() -> trimesh.Trimesh:
        if os.path.exists(os.path.join(mesh_cache_dir, mesh_name)) and use_cache:
            if verbose:
                print(f"Loading mesh {name} from cache {mesh_name} ...")
            mesh = trimesh.load_mesh(os.path.join(mesh_cache_dir, mesh_name))
        else:
            # if verbose:
            if use_cache:
                print(f"{name} does not exist in cache, creating {mesh_name} ...")
            mesh = mesh_gen_fn(cfg)
            mesh.export(os.path.join(mesh_cache_dir, mesh_name))
        return mesh

    return mesh_gen


def check_validity(shape: Tuple[int], indices: Union[np.ndarray, torch.Tensor]):
    """Check if indices are valid for a given shape."""
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)
    if len(shape) == 2:
        is_valid = torch.logical_and(
            torch.logical_and(indices[:, 0] >= 0, indices[:, 0] < shape[0]),
            torch.logical_and(indices[:, 1] >= 0, indices[:, 1] < shape[1]),
        )
    elif len(shape) == 3:
        is_valid = torch.logical_and(
            torch.logical_and(
                torch.logical_and(indices[:, 0] >= 0, indices[:, 0] < shape[0]),
                torch.logical_and(indices[:, 1] >= 0, indices[:, 1] < shape[1]),
            ),
            torch.logical_and(indices[:, 2] >= 0, indices[:, 2] < shape[2]),
        )
    else:
        raise ValueError(f"Invalid shape. shape: {shape}. shape dimension must be 2 or 3.")
    return is_valid


def sample_interpolated_bilinear(
    grid: Union[np.ndarray, torch.Tensor],
    indices: Union[np.ndarray, torch.Tensor],
    invalid_value: float = 0.0,
    round_decimals: int = 5,
):
    """Sample a grid at given indices. If the indices are not integers, interpolate.
    Args:
        grid: (np.ndarray or torch.Tensor) of shape (H, W) or (H, W, D).
        indices: (np.ndarray or torch.Tensor) of shape (N, 2) or (N, 3).
        invalid_value: (float) value to return if the indices are invalid.
        round_decimals: (int) number of decimals to round the indices to.
    Returns:
        (np.ndarray or torch.Tensor) of shape (N,).
    """

    use_pytorch = isinstance(grid, torch.Tensor)

    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    indices = torch.round(indices, decimals=round_decimals)
    # convert the float indices to integer indices
    floor_indices = torch.floor(indices - 0.0).long()
    is_valid = check_validity(grid.shape, floor_indices)
    values = torch.zeros_like(indices[:, 0])
    weights = torch.zeros_like(indices[:, 0])
    for delta in product(*[[0, 1] for _ in range(floor_indices.shape[-1])]):
        delta = torch.tensor(delta, dtype=torch.long).to(floor_indices.device)
        # neighboring_indices.append(floor_indices[:, i])
        idx = floor_indices.clone()
        idx += delta
        # Trilinear interpolation
        w = (1.0 - (indices - idx.float()).abs()).prod(dim=-1)
        # neighboring_indices.append(idx)
        valid = check_validity(grid.shape, idx)
        is_valid = torch.logical_or(is_valid, valid)
        v = torch.ones_like(w) * invalid_value
        v[valid] = grid[[idx[valid, i] for i in range(idx.shape[-1])]].to(v.dtype)
        values[valid] += w[valid] * v[valid]
        weights[valid] += w[valid]
    values /= weights + 1e-6
    values[~is_valid] = invalid_value
    if not use_pytorch:
        values = values.cpu().numpy()
    return values


def sample_interpolated(
    grid: Union[np.ndarray, torch.Tensor],
    indices: Union[np.ndarray, torch.Tensor],
    padding_mode: str = "zeros",
    invalid_value: float = 0.0,
    no_grad: bool = True,
):
    """Sample a grid at given indices. If the indices are not integers, interpolate.
    Args:
        grid: (np.ndarray or torch.Tensor) of shape (B, C, H, W) or (B, C, H, W, D).
        indices: (np.ndarray or torch.Tensor) of shape (B, N, 2) or (B, N, 3).
        invalid_value: (float) value to return if the indices are invalid.
        round_decimals: (int) number of decimals to round the indices to.
    Returns:
        (np.ndarray or torch.Tensor) of shape (N,).
    """

    use_pytorch = isinstance(grid, torch.Tensor)

    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    grid = grid.float()
    indices = indices.float()

    # project the indices to [-1, 1]
    grid_shape = grid.shape
    old_indices = indices.clone()
    if len(grid_shape) == 4:
        indices[..., 0] = old_indices[..., 1] / (grid_shape[-1] - 1) * 2 - 1
        indices[..., 1] = old_indices[..., 0] / (grid_shape[-2] - 1) * 2 - 1
    elif len(grid_shape) == 5:
        indices[..., 0] = old_indices[..., 2] / (grid_shape[-1] - 1) * 2 - 1
        indices[..., 1] = old_indices[..., 1] / (grid_shape[-2] - 1) * 2 - 1
        indices[..., 2] = old_indices[..., 0] / (grid_shape[-3] - 1) * 2 - 1

    if no_grad:
        with torch.no_grad():
            values = F.grid_sample(grid, indices, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    else:
        values = F.grid_sample(grid, indices, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    if invalid_value != 0.0:
        values = torch.where(values == 0.0, torch.tensor(invalid_value), values)
    if not use_pytorch:
        values = values.cpu().numpy()
    return values


def euler_angles_to_rotation_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
    """Convert euler angles to rotation matrix.
    Args:
        roll: (float) rotation around x-axis.
        pitch: (float) rotation around y-axis.
        yaw: (float) rotation around z-axis.
    Returns:
        (np.ndarray) of shape (3, 3).
    """
    roll = np.expand_dims(roll, axis=-1)
    pitch = np.expand_dims(pitch, axis=-1)
    yaw = np.expand_dims(yaw, axis=-1)
    r = Rotation.from_euler("xyz", np.concatenate([roll, pitch, yaw], axis=-1), degrees=False)
    return r.as_matrix()
