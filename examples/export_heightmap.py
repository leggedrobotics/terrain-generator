import trimesh
import pandas as pd
import numpy as np
from terrain_generator.utils.mesh_utils import get_height_array_of_mesh

if __name__ == "__main__":
    mesh = trimesh.load("results/generated_terrain/mesh_0/mesh.obj")
    heightmap = get_height_array_of_mesh(mesh,(40.,40.,40.),800, 0.01, False)
    DF = pd.DataFrame(np.fliplr(np.flipud(heightmap)))
 
    # save the dataframe as a csv file and hdf5 file
    DF.to_csv("results/generated_terrain/mesh_0/heightmap.csv",header=None,index=None)