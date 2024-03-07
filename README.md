# Terrain Generator
This is a automated terrain geneartor tool using [wave function collapse](https://github.com/mxgmn/WaveFunctionCollapse) method. 
![Confined Terrain Generation](doc/confined-terrain-generation.gif)

## Tiling meshes
It checks the connectivity of each mesh parts and connect them.
<p float="left">
  <img src="doc/tiling.gif" width="49%" />
  <img src="doc/different-terrains.gif" width="49%" /> 
</p>



# Install
If you're using conda, create env with the following command.
```bash
conda env create -f environment.yaml
pip install -e .
```

# Usage
To run a testing script run as follows.
```bash
conda activate wfc
python3 examples/generate_with_wfc.py
```
This will first generate all configured meshes and then build different combinations.
Once the mesh is generated, it is stored as cache and reused for the next time.

You can make your own config to generate different terrians.

# Config
TODO
