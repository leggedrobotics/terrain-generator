# Terrain Generator
This is a automated terrain geneartor tool using [wave function collapse](https://github.com/mxgmn/WaveFunctionCollapse) method.

This is used in the paper, [Learning to walk in confined spaces using 3D representation](https://takahiromiki.com/publication-posts/learning-to-walk-in-confined-spaces-using-3d-representation/)

[Project page](https://takahiromiki.com/publication-posts/learning-to-walk-in-confined-spaces-using-3d-representation/), [arxiv](https://arxiv.org/abs/2403.00187), [Youtube](https://youtu.be/QAwBoN55p9I)

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

# Citation
Please cite the following paper if you use this software.

```
@article{miki2024learning,
  title={Learning to walk in confined spaces using 3D representation},
  author={Miki, Takahiro and Lee, Joonho and Wellhausen, Lorenz and Hutter, Marco},
  journal={arXiv preprint arXiv:2403.00187},
  year={2024}
}
```

# Config
TODO
