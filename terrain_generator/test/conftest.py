import pytest


def pytest_addoption(parser):
    parser.addoption("--visualize", action="store_true", help="To enable debug visualizer: bool")
    parser.addoption("--sdf_path", type=str, default="", help="Path to SDF file: str")


@pytest.fixture
def visualize(request):
    return request.config.getoption("--visualize")


@pytest.fixture
def sdf_path(request):
    return request.config.getoption("--sdf_path")
