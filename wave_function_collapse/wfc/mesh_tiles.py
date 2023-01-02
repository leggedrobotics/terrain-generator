import numpy as np
import trimesh

from tiles import Tile, ArrayTile


class MeshTile(ArrayTile):
    def __init__(self, name, array, mesh, edges={}, dimension=2):
        self.mesh = mesh
        super().__init__(name, array, edges, dimension)

    def get_flipped_tile(self, direction):
        # flip array
        mesh = self.mesh.copy()
        if direction == "x":
            # Create the transformation matrix for inverting the mesh in the x-axis
            transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [1, 0, 0])
        elif direction == "y":
            transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 1, 0])
        else:
            raise ValueError(f"Direction {direction} is not defined.")
        # Apply the transformation to the mesh
        mesh.apply_transform(transform)
        tile = super().get_flipped_tile(direction)
        return MeshTile(
            name=tile.name, array=tile.array, mesh=mesh, edges=tile.edges, dimension=self.dimension
        )

    def get_rotated_tile(self, deg):
        if deg not in self.directions.directions:
            raise ValueError(f"Rotation degree {deg} is not defined.")
        transform = trimesh.transformations.rotation_matrix(deg * np.pi / 180, [0, 0, 1])
        mesh = self.mesh.copy()
        mesh = mesh.apply_transform(transform)
        tile = super().get_rotated_tile(deg)
        return MeshTile(
            name=tile.name, array=tile.array, mesh=mesh, edges=tile.edges, dimension=self.dimension
        )

    def get_all_tiles(self, rotations=(), flips=()):
        tiles = [self]
        for flip_direction in flips:
            tiles.append(self.get_flipped_tile(flip_direction))
            for rotation in rotations:
                tiles.append(self.get_flipped_tile(flip_direction).get_rotated_tile(rotation))
        return tiles

    def get_mesh(self):
        return self.mesh
