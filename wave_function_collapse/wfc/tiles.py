import numpy as np

from .wfc import Direction2D, Direction3D


class Tile:
    """Class to manage the tiles."""

    def __init__(self, name, edges={}, rotations=(), dimension=2):
        self.name = name
        self.dimension = dimension
        self.rotations = rotations
        self.edges = edges
        self.directions = Direction2D() if dimension == 2 else Direction3D()

    def get_tile(self):
        return self.name, self.edges, self.rotations

    def get_flipped_tile(self, direction):
        # if direction == "x":
        if direction not in ["x", "y", "z"]:
            raise ValueError(f"Direction {direction} is not defined.")
        new_name = f"{self.name}_{direction}"
        new_edges = {}
        for key, value in self.edges.items():
            new_key = self.directions.flipped_directions[direction][self.directions.base_directions.index(key)]
            if key in self.directions.is_edge_flipped[direction]:
                new_edges[new_key] = value[::-1]
            else:
                new_edges[new_key] = value
        return Tile(name=new_name, edges=new_edges, dimension=self.dimension)


class ArrayTile(Tile):
    """Class to manage the tiles."""

    def __init__(self, name, array, edges={}, rotations=(), dimension=2):
        self.array = array
        self.directions = Direction2D()
        if len(edges) == 0:
            edges = self.create_edges_from_array(array)
        super().__init__(name, edges, rotations, dimension)

    def get_tile(self):
        return self.name, self.edges

    def get_array(self, name=None):
        if name is None:
            return self.array
        for deg in (90, 180, 270):
            if name == f"{self.name}_{deg}":
                a = deg // 90
                return np.rot90(self.array, a)

    def get_flipped_tile(self, direction):
        # flip array
        if direction == "x":
            array = np.flip(self.array, 0)
        elif direction == "y":
            array = np.flip(self.array, 1)
        tile = super().get_flipped_tile(direction)
        return ArrayTile(
            name=tile.name, array=array, edges=tile.edges, rotations=self.rotations, dimension=self.dimension
        )

    def create_edges_from_array(self, array):
        """Create a hash for each edge of the tile."""
        edges = {}
        for direction in self.directions.base_directions:
            if direction == "up":
                edges[direction] = tuple(array[0, :])
            elif direction == "down":
                edges[direction] = tuple(array[-1, :][::-1])
            elif direction == "left":
                edges[direction] = tuple(array[:, 0][::-1])
            elif direction == "right":
                edges[direction] = tuple(array[:, -1])
            else:
                raise ValueError(f"Direction {direction} is not defined.")
        return edges
