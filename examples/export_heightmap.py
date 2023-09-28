import trimesh
import pandas as pd
import numpy as np
import os
from terrain_generator.utils.mesh_utils import get_height_array_of_mesh
from multiprocessing import Pool
import argparse
import shutil
import json
import yaml
import math

def save_to_heightmap(name, save_path, dimension, resolution):
  try:
    mesh = trimesh.load(name)
    print("Loaded mesh")
    heightmap = get_height_array_of_mesh(mesh, (dimension, dimension, dimension), math.ceil(dimension / resolution), resolution, False)
    print("Sampled mesh")
    DF = pd.DataFrame(heightmap)
    print("Converted to DataFrame")

    # save the dataframe as a csv file and hdf5 file
    DF.to_csv(save_path, header=None, index=None)
    print("Saved CSV")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "configs", "mcc_cfg.py"), os.path.join(os.path.dirname(save_path), "terrain_config.py"))
    shutil.copyfile(name, os.path.join(os.path.dirname(save_path), "mesh.obj"))
  except Exception as e:
    print(f"Error processing {name}: {str(e)}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create mesh from configuration")
  parser.add_argument(
      "--mesh_dir", type=str, default="results/generated_terrain", help="Directory to retrieve the mesh files from"
  )
  parser.add_argument(
      "--export_dir", type=str, default="results/generated_terrain/terrains", help="Directory to save the generated heightmap files. Should be an external dir, or empty folder as a subfolder."
  )
  parser.add_argument(
      "--dimension", type=float, default=9, help="Size in meters of terrain (square)"
  )
  parser.add_argument(
      "--resolution", type=float, default=0.01, help="Resolution of the mesh to be exported (in meters)"
  )

  args = parser.parse_args()
  pool = Pool()
  index = 0

  dictionary = {}
  direction_dict = {"90": "R", "180": "S", "270": "L"}

  for root, subdirs, _ in os.walk(args.mesh_dir):
    for subdir in subdirs:
      if os.path.join(root, subdir) in args.export_dir:
        continue
      for _, _, files in os.walk(os.path.join(root, subdir)):
        for fileName in files:
          name = os.path.join(root, subdir, fileName)
          mesh_name = fileName.split("_")
          if len(mesh_name) == 2:
            subdict = {"type": [], "direction": "S", "amplitude": mesh_name[1][0:3]}
          elif len(mesh_name) == 3:
            subdict = {"type": [], "direction": direction_dict.get(mesh_name[1]), "amplitude": mesh_name[2][0:3]}
          save_dir = "terrain_" + str(index)
          index += 1
          type = subdir.split('_', 1)
          subdict["type"] = type[1]
          dictionary[save_dir] = subdict
          save_path = os.path.join(args.export_dir, save_dir, "heightmap.csv")
          if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
          pool.apply_async(save_to_heightmap, args=(name, save_path, args.dimension, args.resolution, ))
    with open(os.path.join(args.export_dir, "metadata.json"), 'w') as json_file:
      json.dump(dictionary, json_file, indent="")
    with open(os.path.join(args.export_dir, "metadata.yaml"), 'w') as yaml_file:
      for key, value in dictionary.items():
        yaml_file.write(f'{key}: {value}\n')
  pool.close()
  pool.join()

