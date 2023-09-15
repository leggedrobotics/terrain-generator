import trimesh
import pandas as pd
import numpy as np
import os
from terrain_generator.utils.mesh_utils import get_height_array_of_mesh
from multiprocessing import Pool
import argparse
import shutil

def save_to_heightmap(name, save_path):
  mesh = trimesh.load(name)
  print("Loaded mesh")
  heightmap = get_height_array_of_mesh(mesh,(9.,9.,10.),900, 0.01, False)
  print("sampled mesh")
  DF = pd.DataFrame(heightmap)
  print("converted to DataFrame")

  # save the dataframe as a csv file and hdf5 file
  DF.to_csv(save_path ,header=None,index=None)
  print("saved CSV")
  shutil.copyfile(os.path.join(os.path.dirname(__file__), "configs","mcc_cfg.py"), os.path.join(os.path.dirname(save_path),"terrain_config.py"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create mesh from configuration")
  parser.add_argument(
      "--mesh_dir", type=str, default="results/generated_terrain", help="Directory to save the generated heightmap files"
  )
  args = parser.parse_args()
  pool = Pool()
  for root, subdirs, _ in os.walk(args.mesh_dir):
    for subdir in subdirs:
      for _, _, files in os.walk(os.path.join(root, subdir)):
        for fileName in files:
          name = os.path.join(root, subdir, fileName)
          _,mesh_name = fileName.split("_", 1)
          save_path = os.path.join(root, subdir, "heightmap_" + mesh_name[0:3] + ".csv")
          pool.apply_async(save_to_heightmap, args=(name, save_path, ))
          # p = Process(target = save_to_heightmap, args=(name, save_path, ))
          # p.start()
  pool.close()
  pool.join()

