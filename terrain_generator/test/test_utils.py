import numpy as np
import torch
import trimesh
import time

from ..utils import sample_interpolated, sample_interpolated_bilinear


def test_interpolated_sampling(visualize=True):

    # 2D array
    array = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).reshape(1, 1, 3, 3)
    points = torch.Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
            [1.5, 1.0],
            [0.1, 0.0],
            [100.0, 0.0],  # Outside
        ]
    ).reshape(1, -1, 1, 2)
    values = sample_interpolated(array, points, invalid_value=0.0).flatten()
    expected_values = torch.Tensor([1.0, 5.0, 3.0, 8.0, 6.5, 1.3, 0.0])
    print("value ", values)
    print("expected value ", expected_values)

    assert torch.allclose(values, expected_values)

    # 3D array
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    array = (
        torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 10, 10], [10, 21, 30], [40, 50, 60]]])
        .reshape(1, 1, 2, 3, 3)
        .to(device)
    )
    points = (
        torch.Tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.5],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [0.10, 0.10, 0.10],
                [2.0, 0.0, 0.0],  # Outside
            ]
        )
        .to(device)
        .reshape(1, -1, 1, 1, 3)
    )
    values = sample_interpolated(array, points, invalid_value=100.0).flatten()
    expected_values = torch.Tensor([1.0, 6.250, 10.0, 2.0, 1.5, 21.0, 7.8750, 2.2710, 100.0]).to(device)
    print("value ", values)
    print("expected value ", expected_values)
    assert torch.allclose(values, expected_values)
    # assert values.device == device

    # numpy
    array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9.0]]]).reshape(1, 1, 3, 3)
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
            [1.5, 1.0],
            [0.1, 0.0],
            [100.0, 0.0],  # Outside
        ]
    ).reshape(1, -1, 1, 2)
    values = sample_interpolated(array, points).flatten()
    expected_values = np.array([1.0, 5.0, 3.0, 8.0, 6.5, 1.3, 0.0])
    assert np.allclose(values, expected_values)

    # Large 2D array
    array = np.arange(400).reshape(1, 1, 20, 20).astype(float)
    x = np.linspace(0, 19, 100)
    y = np.linspace(0, 19, 100)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([yy, xx], axis=-1).reshape(1, 100, 100, 2)
    s = time.time()
    values = sample_interpolated(array, points).flatten()
    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(array.reshape(20, 20))
        plt.show()
        print(points, values)

        plt.imshow(values.reshape(100, 100))
        plt.show()


def test_interpolated_sampling_bilinear(visualize=False):

    # 2D array
    array = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).reshape(3, 3)
    points = torch.Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
            [1.5, 1.0],
            [0.1, 0.0],
            [100.0, 0.0],  # Outside
        ]
    )
    values = sample_interpolated_bilinear(array, points)
    expected_values = torch.Tensor([1.0, 5.0, 3.0, 8.0, 6.5, 1.3, 0.0])

    assert torch.allclose(values, expected_values)

    # 3D array
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    array = (
        torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 10, 10], [10, 21, 30], [40, 50, 60]]])
        .reshape(2, 3, 3)
        .to(device)
    )
    points = torch.Tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.10, 0.10, 0.10],
            [2.0, 0.0, 0.0],  # Outside
        ]
    ).to(device)
    values = sample_interpolated_bilinear(array, points, invalid_value=100)
    expected_values = torch.Tensor([1.0, 6.250, 10.0, 2.0, 1.5, 21.0, 7.8750, 2.2710, 100.0]).to(device)
    assert torch.allclose(values, expected_values)
    assert values.device == device

    # numpy
    array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).reshape(3, 3)
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
            [1.5, 1.0],
            [0.1, 0.0],
            [100.0, 0.0],  # Outside
        ]
    )
    values = sample_interpolated_bilinear(array, points)
    expected_values = np.array([1.0, 5.0, 3.0, 8.0, 6.5, 1.3, 0.0])
    assert np.allclose(values, expected_values)

    # Large 2D array
    array = np.arange(400).reshape(20, 20)
    x = np.linspace(0, 19, 100)
    y = np.linspace(0, 19, 100)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([yy, xx], axis=-1).reshape(-1, 2)
    values = sample_interpolated_bilinear(array, points)
    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(array)
        plt.show()
        print(points, values)

        plt.imshow(values.reshape(100, 100))
        plt.show()
