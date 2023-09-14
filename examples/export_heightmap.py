import trimesh
import pandas as pd
import numpy as np
import os
from terrain_generator.utils.mesh_utils import get_height_array_of_mesh
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mesh from configuration")
    parser.add_argument(
        "--mesh_dir", type=str, default="results/generated_terrain", help="Directory to save the generated heightmap files"
    )
    for root, subdirs,_ in os.walk(args.mesh_dir):
        for subdir in subdirs:
            mesh = trimesh.load(os.path.join(root, subdir, "mesh.obj"))
            print("Loaded mesh")
            heightmap = get_height_array_of_mesh(mesh,(40.,40.,40.),800, 0.01, False)
            print("sampled mesh")
            DF = pd.DataFrame(heightmap)
            print("converted to DataFrame")
 
            # save the dataframe as a csv file and hdf5 file
            DF.to_csv(os.path.join(root, subdir, "heightmap.csv"),header=None,index=None)
            print("saved CSV")
