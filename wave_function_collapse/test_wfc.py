from wfc import WFC


def test_get_neighbours():

    # Test 2D
    connections = {
        0: {(-1, 0): (0, 1), (0, -1): (0), (1, 0): (0, 1), (0, 1): (0, 1)},  # Mountain
        1: {(-1, 0): (0, 1, 2), (0, -1): (0, 1), (1, 0): (0, 1, 2), (0, 1): (0, 1, 2)},  # Sand
        2: {(-1, 0): (1, 2), (0, -1): (2), (1, 0): (2), (0, 1): (2)},  # Water
    }
    wfc = WFC(3, connections, (10, 10))
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
    wfc = WFC(3, connections, (10, 10, 10), dimensions=3)
    n, d = wfc._get_neighbours((2, 2, 2))
    assert n.dtype == int
    assert n.shape == (6, 3)
    assert sorted(n.tolist()) == sorted([[1, 2, 2], [3, 2, 2], [2, 1, 2], [2, 3, 2], [2, 2, 1], [2, 2, 3]])


def test_2d_case():
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


def test_3d_case():
    directions = [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    connections = {
        0: {directions[i]: (0, 1, 2) for i in range(6)},
        1: {directions[i]: (0, 1, 2) for i in range(6)},
        2: {directions[i]: (0, 2) for i in range(6)},
    }
    wfc = WFC(3, connections, (10, 10, 3), dimensions=3)
    n, d = wfc._get_neighbours((9, 9, 1))
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
