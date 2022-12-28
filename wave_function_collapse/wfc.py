import random
from threading import local
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal, Tuple
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
        non_possible_tiles = [np.setdiff1d(np.arange(self.n_tiles), pt) for pt in possible_tiles]
        for neighbor, tiles in zip(neighbours, non_possible_tiles):
            self.valid[(tiles,) + tuple(neighbor)] = False

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
            # print("wave\n", self.wave)
            idx = self.collapse(entropy)
            # idx = np.unravel_index(np.argmin(entropy), entropy.shape)
            if entropy[tuple(idx)] == self.n_tiles + 1:
                break
            if entropy[tuple(idx)] == 0:
                raise ValueError("No valid tiles for the given constraints.")
            self.observe(idx)
            self.history.append(self.wave.copy())

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
        # up: tuple = (0, -1)
        # down: tuple = (0, 1)
        # left: tuple = (-1, 0)
        # right: tuple = (1, 0)
        directions: tuple = ("up", "left", "down", "right")

        def rotate(self, direction, deg):
            if deg == 90:
                new_directions = np.roll(np.array(self.directions), -1)
            elif deg == 180:
                new_directions = np.roll(np.array(self.directions), -2)
            elif deg == 270:
                new_directions = np.roll(np.array(self.directions), -3)
            else:
                raise ValueError("deg must be 90, 180 or 270.")
            return new_directions[self.directions.index(direction)]

        def flip(self, direction, axis):
            if axis == "x":
                new_directions = np.array(self.directions)[[2, 1, 0, 3]]
            elif axis == "y":
                new_directions = np.array(self.directions)[[0, 3, 2, 1]]
            else:
                raise ValueError("axis must be x or y.")
            return new_directions[self.directions.index(direction)]

        def get_name(self, direction):
            for k, v in self.__dict__.items():
                if v == direction:
                    return k

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

        def rotate(self, direction, deg):
            if deg == 90:
                new_directions = self.directions[:2] + tuple(np.roll(np.array(self.directions[2:]), -1))
            elif deg == 180:
                new_directions = self.directions[:2] + tuple(np.roll(np.array(self.directions[2:]), -2))
            elif deg == 270:
                new_directions = self.directions[:2] + tuple(np.roll(np.array(self.directions[2:]), -3))
            else:
                raise ValueError("deg must be 90, 180 or 270.")
            return new_directions[self.directions.index(direction)]

        def flip(self, direction, axis):
            if axis == "x":
                new_directions = np.array(self.directions)[[0, 1, 2, 3, 5, 4]]
            elif axis == "y":
                new_directions = np.array(self.directions)[[0, 1, 3, 2, 4, 5]]
            elif axis == "z":
                new_directions = np.array(self.directions)[[1, 0, 2, 3, 4, 5]]
            else:
                raise ValueError("axis must be x or y or z.")
            return new_directions[self.directions.index(direction)]

        def get_name(self, direction):
            for k, v in self.__dict__.items():
                if v == direction:
                    return k

    direction_3d = Direction3D()

    def __init__(self, dimensions=2):
        self.connections = {}
        self.names = []
        self.connection_types = {}
        self.edge_types = {}  # dict of edges. key: tile_name, value: dict of edges
        self.edges = {}  # dict of edges for each tile key: edge_type, value: (direction, tile_name)
        if dimensions == 2:
            self.directions = self.direction_2d
        elif dimensions == 3:
            self.directions = self.direction_3d
        else:
            raise ValueError("Only 2D and 3D are supported.")

    def register_tile(
        self,
        name: str,
        edge_types: dict,
        allow_rotation_deg: Tuple[int, ...] = (90, 180, 270),
        reflection_dir: Tuple[str, ...] = ("x", "y"),
    ):

        self._register_tile(name, edge_types)

        if len(allow_rotation_deg) > 0:
            self._register_rotations(name, edge_types, allow_rotation_deg)

        if len(reflection_dir) > 0:
            self._register_reflection(name, edge_types, reflection_dir)

    def _register_tile(self, name: str, edge_types: dict):
        self.names.append(name)
        edge_types_tuple = {}
        for direction, edge in edge_types.items():
            d = self._to_tuple(direction)
            if edge not in self.edges:
                self.edges[edge] = []
            self.edges[edge].append((d, name))
            edge_types_tuple[d] = edge
        self.edge_types[name] = edge_types_tuple

    def _register_rotations(self, name: str, edge_types: dict, allow_rotation_deg: Tuple[int, ...]):
        # print("name ", name)
        # print("edge_types ", edge_types)
        # 90 degree rotation
        # allow_rotation_deg = [90, 180, 270]
        for deg in allow_rotation_deg:
            rotated_edge_types = {}
            rotated_name = f"{name}_{deg}"
            for direction, edge in edge_types.items():
                # print("direction ", direction)
                d = self.directions.rotate(direction, deg)
                # print("new direction", d)
                rotated_edge_types[d] = edge

            # print("rotated_name ", rotated_name)
            # print("rotated_edge_types ", rotated_edge_types)
            self._register_tile(rotated_name, rotated_edge_types)

    def _register_reflection(self, name: str, edge_types: dict, reflection_dir: Tuple[str, ...]):
        for axis in reflection_dir:
            # print("axis ", axis)
            reflected_edge_types = {}
            reflected_name = f"{name}_{axis}"
            for direction, edge in edge_types.items():
                d = self.directions.flip(direction, axis)
                # print("new direction", d)
                reflected_edge_types[d] = edge
                # reflected_edge_types[d] = edge[::-1]
            self._register_tile(reflected_name, reflected_edge_types)

    def compute_connection_dict(self):
        connections = {}
        readable_connections = {}
        for name in self.names:
            # print("name ", name)
            edge_types = self.edge_types[name]
            # print("edge_types ", edge_types)
            local_connections = {}
            local_readable_connections = {}
            # print("edges ", self.edges)
            for direction_1, edge_type in edge_types.items():
                local_connections[direction_1] = set()
                local_readable_connections[self.directions.get_name(direction_1)] = set()
                # print("direction ", direction_1)
                # print("edge_type ", edge_type)
                for direction_2, name_2 in self.edges[edge_type]:
                    # print("opponent ", name_2)
                    if (np.array(direction_1) + np.array(direction_2) == 0).all():
                        local_connections[direction_1].add(name_2)
                        local_readable_connections[self.directions.get_name(direction_1)].add(name_2)
            # print("local_connections", local_connections)
            connections[name] = local_connections
            readable_connections[name] = local_readable_connections
        # print("connection ", connections)
        # self.connections = connections
        self.readable_connections = readable_connections
        return self._replace_name_with_number(connections)

    def _to_tuple(self, direction):
        return getattr(self.directions, direction)

    # def _get_direction_vector(self, direction):
    #     return np.array(getattr(self.directions, direction))

    def _replace_name_with_number(self, d: dict):
        """Replace the names of the tiles with numbers. Recursively."""
        # if there are string which is in self.names, replace it with the index of self.names
        new_d = {}
        for key, value in d.items():
            if isinstance(key, str) and key in self.names:
                new_d[self.names.index(key)] = value
            else:
                new_d[key] = value
        d = new_d
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self._replace_name_with_number(value)
            elif isinstance(value, str):
                if value in self.names:
                    d[key] = self.names.index(value)
            elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
                value = list(value)
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        if item in self.names:
                            value[i] = self.names.index(item)
                d[key] = tuple(sorted(value))
            else:
                print("type of value ", type(value))
                raise ValueError("Unknown type in the connection dict.")
        return d


if __name__ == "__main__":

    cm = ConnectionManager(dimensions=2)
    cm.register_tile(
        "A", {"up": "a", "down": "a", "left": "b", "right": "b"}, allow_rotation_deg=(90,), reflection_dir=()
    )
    cm.register_tile(
        "B",
        {"up": "a", "down": "b", "left": "b", "right": "a"},
        allow_rotation_deg=(90, 180, 270),
        reflection_dir=("x", "y"),
    )
    cm.register_tile("C", {"up": "a", "down": "a", "left": "a", "right": "a"}, allow_rotation_deg=(), reflection_dir=())
    # cm.register_tile("A", {"up": "a", "down": "a", "left": "a", "right": "a", "front": "a", "back": "a"})
    # cm.register_tile("B", {"up": "a", "down": "a", "left": "b", "right": "b", "front": "b", "back": "b"})
    # cm.register_tile("C", {"up": "a", "down": "a", "left": "c", "right": "c", "front": "c", "back": "c"})
    # cm.register_connection_rule("a", "b", True)
    connections = cm.compute_connection_dict()
    print("connections ", connections)
    print("readable_connections ", cm.readable_connections)
    print("names ", cm.names)

    # connections = {
    #     0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
    #     1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
    #     2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    # }
    wfc = WFC(len(cm.names), connections, [40, 40])
    # n, d = wfc._get_neighbours((9, 9))
    # print("Neighbours:", n, n.dtype, d, d.dtype)
    wfc.init_randomly()
    wave = wfc.solve()

    print("wave ", wave)
    # for w in wfc.history:
    #     print(w)
    #
    import matplotlib.pyplot as plt

    # from matplotlib.colors import LinearSegmentedColormap

    def rotate(tile, angle):
        a = angle // 90
        return np.rot90(tile, a)

    def flip(tile, axis):
        if axis == "x":
            a = 0
        elif axis == "y":
            a = 1
        return np.flip(tile, a)

    tiles = {}
    tiles["A"] = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    tiles["B"] = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
    tiles["C"] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("A", tiles["A"])
    print("A_90", rotate(tiles["A"], 90))
    print("A_180", rotate(tiles["A"], 180))
    print("A_270", rotate(tiles["A"], 270))
    print("A_x", flip(tiles["A"], "x"))
    print("A_y", flip(tiles["A"], "y"))

    rotations = [90, 180, 270]
    flips = ["x", "y"]

    names = list(tiles.keys())

    for name in names:
        for r in rotations:
            tiles[f"{name}_{r}"] = rotate(tiles[name], r)
        for f in flips:
            tiles[f"{name}_{f}"] = flip(tiles[name], f)

    # print("A", tiles["A"])
    # print("A_90", rotate(tiles["A"], 90))
    # print("tiles", tiles)
    img = np.zeros((wfc.shape[0] * 3, wfc.shape[1] * 3))
    print("img ", img.shape)
    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            tile = tiles[cm.names[wave[y, x]]]
            # print("name ", cm.names[wave[y, x]])
            # print("tile ", tile)
            img[y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3] = tile
    print("img \n", img[0:9, 0:9])

    plt.imshow(img)
    plt.show()

    # Define the colors and values for the custom colormap
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # values = [0, 1, 2]
    #
    # # Create the custom colormap
    # cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(values))
    #
    # # Use imshow to display the array with the custom colormap
    # plt.imshow(wave, cmap=cmap)

    # Show the plot
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
