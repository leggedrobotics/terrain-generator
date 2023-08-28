#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from ..wfc.wfc import WFCCore, ConnectionManager


def test_get_neighbours():

    # Test 2D
    connections = {
        0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
        1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
        2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    }
    wfc = WFCCore(3, connections, (10, 10))
    n, d = wfc._get_neighbours((2, 2))
    assert n.dtype == int
    assert n.shape == (4, 2)
    assert sorted(n.tolist()) == sorted([[1, 2], [3, 2], [2, 1], [2, 3]])

    # Test 3D
    directions = [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    connections = {
        0: {directions[i]: (i) for i in range(6)},
        1: {directions[i]: (i) for i in range(6)},
        2: {directions[i]: (i) for i in range(6)},
    }
    wfc = WFCCore(3, connections, (10, 10, 10), dimensions=3)
    n, d = wfc._get_neighbours((2, 2, 2))
    assert n.dtype == int
    assert n.shape == (6, 3)
    assert sorted(n.tolist()) == sorted([[1, 2, 2], [3, 2, 2], [2, 1, 2], [2, 3, 2], [2, 2, 1], [2, 2, 3]])


def test_2d_case(visualize=False):
    connections = {
        0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
        1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
        2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    }
    wfc = WFCCore(3, connections, (20, 20))
    n, d = wfc._get_neighbours((9, 9))
    wfc.init_randomly()
    wfc.solve()
    wave = wfc.wave.wave

    if visualize:
        print("Neighbours:", n, n.dtype, d, d.dtype)
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


def test_3d_case(visualize=False):
    directions = [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    connections = {
        0: {directions[i]: (0, 1, 2) for i in range(6)},
        1: {directions[i]: (0, 1, 2) for i in range(6)},
        2: {directions[i]: (0, 2) for i in range(6)},
    }
    wfc = WFCCore(3, connections, [10, 10, 3], dimensions=3)
    n, d = wfc._get_neighbours((9, 9, 1))
    wfc.init_randomly()
    wfc.solve()
    wave = wfc.wave.wave
    if visualize:
        print("Neighbours:", n, n.dtype, d, d.dtype)
        print("wave ", wave)


def test_connections():
    # cm = ConnectionManager(load_from_cache=False)
    # cm.register_tile("tile_1", {"up": "abc", "down": "bbb", "left": "ccc", "right": "ddd"})
    # cm.register_tile("tile_2", {"up": "abc", "down": "bbb", "left": "ddd", "right": "ccc"})
    # cm.register_tile("tile_3", {"up": "bbb", "down": "cba", "left": "ccc", "right": "ddd"})
    # cm.register_tile("tile_4", {"up": "bbb", "down": "cba", "left": "ddd", "right": "ccc"})
    #
    # d = cm.get_connection_dict()
    # print("d = ", d)
    # # for k, v in d.items():
    # #     # print(k, v)
    # #     for kk, vv in v.items():
    # #         print(kk, vv)
    # assert d == {
    #     0: {(-1, 0): {2, 3}, (1, 0): {2, 3}, (0, -1): {1, 3}, (0, 1): {1, 3}},
    #     1: {(-1, 0): {2, 3}, (1, 0): {2, 3}, (0, -1): {0, 2}, (0, 1): {0, 2}},
    #     2: {(-1, 0): {0, 1}, (1, 0): {0, 1}, (0, -1): {1, 3}, (0, 1): {1, 3}},
    #     3: {(-1, 0): {0, 1}, (1, 0): {0, 1}, (0, -1): {0, 2}, (0, 1): {0, 2}},
    # }

    cm = ConnectionManager(load_from_cache=False)
    cm.register_tile("tile_1", {"up": (0, 1, 2), "down": (1, 1, 1), "left": (2, 2, 2), "right": (3, 3, 3)})
    cm.register_tile("tile_2", {"up": (0, 1, 2), "down": (1, 1, 1), "left": (3, 3, 3), "right": (2, 2, 2)})
    cm.register_tile("tile_3", {"up": (1, 1, 1), "down": (2, 1, 0), "left": (2, 2, 2), "right": (3, 3, 3)})
    cm.register_tile("tile_4", {"up": (1, 1, 1), "down": (2, 1, 0), "left": (3, 3, 3), "right": (2, 2, 2)})

    d = cm.get_connection_dict()
    # print("d = ", d)
    # for k, v in d.items():
    #     # print(k, v)
    #     for kk, vv in v.items():
    #         print(kk, vv)
    # assert d == {
    #     0: {(-1, 0): {2, 3}, (1, 0): {2, 3}, (0, -1): {1, 3}, (0, 1): {1, 3}},
    #     1: {(-1, 0): {2, 3}, (1, 0): {2, 3}, (0, -1): {0, 2}, (0, 1): {0, 2}},
    #     2: {(-1, 0): {0, 1}, (1, 0): {0, 1}, (0, -1): {1, 3}, (0, 1): {1, 3}},
    #     3: {(-1, 0): {0, 1}, (1, 0): {0, 1}, (0, -1): {0, 2}, (0, 1): {0, 2}},
    # }

    wfc = WFCCore(
        len(cm.names),
        d,
        [10, 10],
        dimensions=2,
    )
    # print("Start solving...")
    # if len(init_tiles) > 0:
    #     # print("init ", init_args)
    #     for (name, index) in init_tiles:
    #         tile_id = self.cm.names.index(name)
    #         # print("idx ", idx)
    #         wfc.init(index, tile_id)
    # else:
    wfc.init_randomly()
    wave = wfc.solve()
    # print("wave ", wave)

    # connections = {
    #     0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
    #     1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
    #     2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    # }

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
