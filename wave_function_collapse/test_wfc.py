from wfc import WFC


def test_get_neighbours():

    # Test 2D
    wfc = WFC([1, 2, 3], (10, 10))
    n = wfc._get_neighbours((2, 2))
    assert n.dtype == int
    assert n.shape == (4, 2)
    assert sorted(n.tolist()) == sorted([[1, 2], [3, 2], [2, 1], [2, 3]])

    # Test 3D
    wfc = WFC([1, 2, 3], (10, 10, 10), dimensions=3)
    n = wfc._get_neighbours((2, 2, 2))
    assert n.dtype == int
    assert n.shape == (6, 3)
    assert sorted(n.tolist()) == sorted([[1, 2, 2], [3, 2, 2], [2, 1, 2], [2, 3, 2], [2, 2, 1], [2, 2, 3]])
