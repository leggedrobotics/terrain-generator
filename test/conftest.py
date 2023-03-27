import pytest


def pytest_addoption(parser):
    parser.addoption("--visualize", action="store_true", help="To enable debug visualizer: bool")


@pytest.fixture
def visualize(request):
    return request.config.getoption("--visualize")


