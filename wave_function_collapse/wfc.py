import random
from threading import local
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal
import itertools


class WFC:
    """Wave Function Collapse algorithm implementation."""

    def __init__(self, n_tiles: int, connections: dict, shape: list, dimensions: int = 2):
        """Initialize the WFC algorithm.
        Args:
            n_tiles: number of tiles
            connections: dictionary of connections for each tile
            shape: shape of the wave
            dimensions: number of dimensions of the wave
        """

        self.n_tiles = n_tiles
        self.connections = connections
        self.shape = shape
        self.dimensions = dimensions
        self.wave = np.zeros(shape, dtype=np.int32)  # first dimension is for the tile, second is for the orientation
        self.valid = np.ones((self.n_tiles, *shape), dtype=bool)
        self.is_collapsed = np.zeros(shape, dtype=bool)
        self.new_idx = None
        self.history = []

    def _get_neighbours(self, idx):
        """Get the neighbours of a given tile."""
        neighbours = np.tile(idx, (2 * self.dimensions, 1))
        d_idx = np.vstack([np.eye(self.dimensions, dtype=int), -np.eye(self.dimensions, dtype=int)])
        neighbours += d_idx
        # Remove out of bounds neighbours
        neighbours = neighbours[np.all(neighbours >= 0, axis=1) & np.all(neighbours < self.shape, axis=1)]
        directions = neighbours - idx
        return neighbours, directions

    def _get_possible_tiles(self, tile_id, directions):
        """Get the constraints of a given tile.
        Based on the tile's connections and the directions of the neighbours."""
        possible_tiles = []
        for direction in directions:
            d = tuple(direction)
            possible_tiles.append(self.connections[tile_id][d])
        return possible_tiles

    def _update_wave(self, idx: np.ndarray, tile_id: int):
        self.wave[tuple(idx)] = tile_id
        self.is_collapsed[tuple(idx)] = True
        self.valid[(slice(None),) + tuple(idx)] = False
        self.valid[(tile_id,) + tuple(idx)] = True
        self._update_validity(idx, tile_id)

    def _update_validity(self, new_idx: np.ndarray, tile_id: int):
        neighbours, directions = self._get_neighbours(new_idx)
        possible_tiles = self._get_possible_tiles(tile_id, directions)
        for neighbor, tiles in zip(neighbours, possible_tiles):
            self.valid[(slice(None),) + tuple(neighbor)] = False
            self.valid[(tiles,) + tuple(neighbor)] = True

    def _random_collapse(self, entropy):
        """Choose the indices of the tiles to be observed."""
        indices = np.argwhere(entropy == np.min(entropy))
        return np.array(random.choice(indices))

    def collapse(self, entropy):
        """Collapse the wave."""
        # Choose a tile with lowest entropy. If there are multiple, choose randomly.
        return self._random_collapse(entropy)

    def init_randomly(self):
        """Initialize the wave randomly."""
        idx = np.random.randint(0, self.shape, self.dimensions)
        tile_id = np.random.randint(0, self.n_tiles)
        self._update_wave(idx, tile_id)
        self.new_idx = idx

    def observe(self, idx):
        """Observe a tile."""
        tile_id = np.random.choice(np.arange(self.n_tiles)[self.valid[(slice(None),) + tuple(idx)]])
        self._update_wave(idx, tile_id)

    def solve(self):
        """Solve the WFC problem."""
        while True:
            # Find a tile with lowest entropy
            entropy = np.sum(self.valid, axis=0)
            entropy[self.is_collapsed] = self.n_tiles + 1
            # print("entropy\n", entropy)
            idx = self.collapse(entropy)
            # idx = np.unravel_index(np.argmin(entropy), entropy.shape)
            if entropy[tuple(idx)] == self.n_tiles + 1:
                break
            if entropy[tuple(idx)] == 0:
                raise ValueError("No valid tiles for the given constraints.")
            self.observe(idx)
            self.history.append(self.wave)

        return self.wave
        # print("wave ", self.wave)


class ConnectionManager:
    """Class to manage the connections between tiles."""

    @dataclass
    class Direction2D:
        up: tuple = (-1, 0)
        down: tuple = (1, 0)
        left: tuple = (0, -1)
        right: tuple = (0, 1)
        directions: tuple = ("up", "down", "left", "right")

    direction_2d = Direction2D()

    @dataclass
    class Direction3D:
        up: tuple = (0, 0, 1)
        down: tuple = (0, 0, -1)
        left: tuple = (0, -1, 0)
        right: tuple = (0, 1, 0)
        front: tuple = (-1, 0, 0)
        back: tuple = (1, 0, 0)
        directions: tuple = ("up", "down", "left", "right", "front", "back")

    direction_3d = Direction3D()

    def __init__(self, dimensions=2):
        self.connections = {}
        self.names = []
        self.connection_types = {}
        self.edge_types = {}
        self.edges = {}
        if dimensions == 2:
            self.directions = self.direction_2d
        elif dimensions == 3:
            self.directions = self.direction_3d
        else:
            raise ValueError("Only 2D and 3D are supported.")

    def register_tile(self, name: str, edge_types: dict, allow_rotation: bool = True, reflection_dir: tuple = None):
        self.names.append(name)
        edge_types_tuple = {}
        for direction, edge in edge_types.items():
            d = self._to_tuple(direction)
            if edge not in self.edges:
                self.edges[edge] = []
            self.edges[edge].append((d, name))
            edge_types_tuple[d] = edge
        self.edge_types[name] = edge_types_tuple

        if allow_rotation:
            self._register_rotations(name, edge_types)

        if reflection_dir is not None:
            self._register_reflection(name, edge_types, reflection_dir)

    def _register_rotations(self, name: str, edge_types: dict):
        pass

    def _register_reflection(self, name: str, edge_types: dict, reflection_dir: tuple):
        pass

    def compute_connection_dict(self):
        connections = {}
        for name in self.names:
            # print("name ", name)
            edge_types = self.edge_types[name]
            print("edge_types ", edge_types)
            local_connections = {}
            # print("edges ", self.edges)
            for direction_1, edge_type in edge_types.items():
                local_connections[direction_1] = set()
                print("direction ", direction_1)
                # print("edge_type ", edge_type)
                for direction_2, name_2 in self.edges[edge_type]:
                    # print("direction 2", direction_2)
                    # print("opponent ", name_2)
                    if (np.array(direction_1) + np.array(direction_2)).all() == 0:
                        # print("connection found")
                        local_connections[direction_1].add(name_2)
            # print("local_connections", local_connections)
            connections[name] = local_connections
        # print("connection ", connections)
        self.connections = connections
        return self._replace_name_with_number(connections)

    def _to_tuple(self, direction):
        return getattr(self.directions, direction)

    # def _get_direction_vector(self, direction):
    #     return np.array(getattr(self.directions, direction))

    def _replace_name_with_number(self, d: dict):
        """Replace the names of the tiles with numbers. Recursively."""
        # if there are string which is in self.names, replace it with the index of self.names
        for key, value in d.items():
            if isinstance(key, str) and key in self.names:
                d[self.names.index(key)] = d.pop(key)
            if isinstance(value, dict):
                self._replace_name_with_number(value)
            elif isinstance(value, str):
                if value in self.names:
                    d[key] = self.names.index(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._replace_name_with_number(item)
                    elif isinstance(item, str):
                        if item in self.names:
                            value[i] = self.names.index(item)
                d[key] = tuple(sorted(value))
            elif isinstance(value, set):
                value = list(value)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._replace_name_with_number(item)
                    elif isinstance(item, str):
                        if item in self.names:
                            value[i] = self.names.index(item)
                d[key] = tuple(sorted(value))
            else:
                raise ValueError("Unknown type in the connection dict.")
        return d


if __name__ == "__main__":

    cm = ConnectionManager()
    cm.register_tile("A", {"up": "a", "down": "a", "left": "a", "right": "a"})
    cm.register_tile("B", {"up": "a", "down": "a", "left": "b", "right": "b"})
    cm.register_tile("C", {"up": "a", "down": "a", "left": "c", "right": "c"})
    # cm.register_connection_rule("a", "b", True)
    connections = cm.compute_connection_dict()
    print("connections ", connections)

    # connections = {
    #     0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
    #     1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
    #     2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    # }
    # wfc = WFC(3, connections, (20, 20))
    # n, d = wfc._get_neighbours((9, 9))
    # print("Neighbours:", n, n.dtype, d, d.dtype)
    # wfc.init_randomly()
    # wave = wfc.solve()
    #
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LinearSegmentedColormap
    #
    # # Define the colors and values for the custom colormap
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # values = [0, 1, 2]
    #
    # # Create the custom colormap
    # cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(values))
    #
    # # Use imshow to display the array with the custom colormap
    # plt.imshow(wave, cmap=cmap)
    #
    # # Show the plot
    # plt.show()
    #
    # directions = [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # connections = {
    #     0: {directions[i]: (0, 1, 2) for i in range(6)},
    #     1: {directions[i]: (0, 1, 2) for i in range(6)},
    #     2: {directions[i]: (0, 2) for i in range(6)},
    # }
    # wfc = WFC(3, connections, (10, 10, 3), dimensions=3)
    # n, d = wfc._get_neighbours((9, 9, 1))
    # print("Neighbours:", n, n.dtype, d, d.dtype)
    # wfc.init_randomly()
    # wave = wfc.solve()
    #
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LinearSegmentedColormap
    #
    # # Define the colors and values for the custom colormap
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # values = [0, 1, 2]
    #
    # # Create the custom colormap
    # cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(values))
    #
    # # Use imshow to display the array with the custom colormap
    # plt.imshow(wave, cmap=cmap)
    #
    # # Show the plot
    # plt.show()
