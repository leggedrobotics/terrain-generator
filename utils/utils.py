import os
import numpy as np
from pyglet.window.key import P
import trimesh
from typing import Callable, Any, Optional
from dataclasses import asdict, is_dataclass
import open3d as o3d
import copy


ENGINE = "blender"
CACHE_DIR = "__mesh_cache__"
# ENGINE = "scad"


def cfg_to_hash(cfg, exclude_keys=["weight", "load_from_cache"]):
    """MD5 hash of a config."""
    import hashlib
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # if isinstance(obj, tuple):
            #     return str(list(obj))
            # if isinstance(obj, dict):
            #     return self.default(obj)
            return json.JSONEncoder.default(self, obj)

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
    os.makedirs(CACHE_DIR, exist_ok=True)
    if hasattr(cfg, "name"):
        name = cfg.name
    else:
        name = ""

    def mesh_gen() -> trimesh.Trimesh:
        if os.path.exists(os.path.join(CACHE_DIR, code + ".obj")) and use_cache:
            if verbose:
                print(f"Loading mesh {name} from cache {code}.obj ...")
            mesh = trimesh.load_mesh(os.path.join(CACHE_DIR, code + ".obj"))
        else:
            # if verbose:
            print(f"Not loading {name} from cache, creating {code}.obj ...")
            mesh = mesh_gen_fn(cfg)
            mesh.export(os.path.join(CACHE_DIR, code + ".obj"))
        return mesh

    return mesh_gen
