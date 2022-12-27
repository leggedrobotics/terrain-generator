import random
import numpy as np
import numpy.typing as npt
from typing import Literal


class WFC:
    """Wave Function Collapse algorithm implementation."""

    def __init__(self, n_tiles, connections, shape, dimensions=2):
        # self.tiles = tiles
        self.n_tiles = n_tiles
        self.connections = connections
        self.shape = shape
        self.dimensions = dimensions
        self.wave = np.zeros(shape, dtype=np.int32)  # first dimension is for the tile, second is for the orientation
        self.valid = np.ones((self.n_tiles, *shape), dtype=bool)
        self.is_collapsed = np.zeros(shape, dtype=bool)
        self.new_idx = None

    def _get_neighbours(self, idx):
        """Get the neighbours of a given tile."""
        neighbours = np.tile(idx, (2 * self.dimensions, 1))
        d_idx = np.vstack([np.eye(self.dimensions, dtype=int), -np.eye(self.dimensions, dtype=int)])
        neighbours += d_idx
        # Remove out of bounds neighbours
        neighbours = neighbours[np.all(neighbours >= 0, axis=1) & np.all(neighbours < self.shape, axis=1)]
        directions = neighbours - idx
        return neighbours, directions

    def _get_constraints(self, tile_id, directions):
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
        # print("neighbours", neighbours)
        # print("directions", directions)
        possible_tiles = self._get_constraints(tile_id, directions)
        for neighbor, direction, tiles in zip(neighbours, directions, possible_tiles):
            # print("neighbor", neighbor)
            # print("direction", direction)
            # print("tiles", tiles)
            # print("valid", self.valid)
            self.valid[(slice(None),) + tuple(neighbor)] = False
            # print("valid", self.valid)
            self.valid[(tiles,) + tuple(neighbor)] = True
            # print("valid", self.valid)
        # print("possible_tiles", possible_tiles)

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
        self.propagate(self.new_idx)

    def propagate(self, new_idx):
        """Propagate the constraints."""
        pass
        #
        # valid_tiles = self._get_constraints(new_idx)
        # neighbours = self._get_neighbours(new_idx)

    def observe(self, idx):
        """Observe a tile."""
        print("valid tiles ", np.arange(self.n_tiles)[self.valid[(slice(None),) + tuple(idx)]])
        tile_id = np.random.choice(np.arange(self.n_tiles)[self.valid[(slice(None),) + tuple(idx)]])
        print("tile_id ", tile_id)
        self._update_wave(idx, tile_id)
        # self.wave[tuple(idx)] = tile_id
        # self.valid[(slice(None),) + tuple(idx)] = False
        # self.valid[(tile_id,) + tuple(idx)] = True
        # self.new_idx = idx
        # self.propagate(self.new_idx)

    def solve(self):
        """Solve the WFC problem."""
        while True:
            # Find a tile with lowest entropy
            entropy = np.sum(self.valid, axis=0)
            entropy[self.is_collapsed] = self.n_tiles + 1
            print("entropy\n", entropy)
            idx = self.collapse(entropy)
            # idx = np.unravel_index(np.argmin(entropy), entropy.shape)
            if entropy[tuple(idx)] == self.n_tiles + 1:
                break
            if entropy[tuple(idx)] == 0:
                raise ValueError("No valid tiles for the given constraints.")
            self.observe(idx)
            print("wave ", self.wave)
            # self.propagate(idx)
            # break


if __name__ == "__main__":
    connections = {
        0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
        1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
        2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    }
    wfc = WFC(3, connections, (20, 20))
    n, d = wfc._get_neighbours((9, 9))
    print("Neighbours:", n, n.dtype, d, d.dtype)
    wfc.init_randomly()
    wfc.solve()
    wave = wfc.wave

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Define the colors and values for the custom colormap
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    values = [0, 1, 2]

    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(values))

    # Use imshow to display the array with the custom colormap
    plt.imshow(wave, cmap=cmap)

    # Show the plot
    plt.show()
