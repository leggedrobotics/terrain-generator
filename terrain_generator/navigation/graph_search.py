#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_height_array_of_mesh


def cost_function(G, source, target):
    pass


class MeshNavigation(object):
    """Class for navigation based on terrain mesh"""

    def __init__(self, mesh):
        self.mesh = mesh
        self.G = self.create_graph_from_mesh(mesh)

    def create_graph_from_mesh(self, mesh, cost_function=None):
        """Create a graph with costs from a mesh"""
        bbox = mesh.bounds


if __name__ == "__main__":
    import numpy as np
    import networkx as nx
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse.csgraph import shortest_path

    size = (20, 20)

    # Define cost map with random costs
    # cost_map = np.random.randint(1, 10, size=size)
    cost_map = np.ones(size)

    cost_map[5, :15] = 100.0
    cost_map[10, 5:] = 100.0
    cost_map[15, :15] = 100.0

    print("Cost map:")
    print(cost_map)

    # Generate grid graph
    G = nx.grid_2d_graph(*cost_map.shape)
    print("G ", G)
    print("G.nodes():", G.nodes())
    # Add cost map to edges
    for (u, v) in G.edges():
        print("u, v: ", u, v)
        cost = cost_map[u[0], u[1]] + cost_map[v[0], v[1]]
        print("cost: ", cost)
        G[u][v]["weight"] = cost

    # Compute adjacency matrix
    adj_mtx = nx.adjacency_matrix(G)
    print("adj_mtx", adj_mtx, adj_mtx.shape)

    graph = csr_matrix(adj_mtx)

    # Compute shortest path distances using Dijkstra's algorith
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)

    # dist_mtx, _ = dijkstra(csgraph=adj_mtx, directed=False, return_predecessors=False)

    # Print results
    print("Distance matrix:")
    print(dist_matrix, dist_matrix.shape)
    print("d(10, 25) = ", dist_matrix[10, 25])
    nodes_list = np.array(list(G.nodes()))
    print("node (10, 25) = ", nodes_list[10], nodes_list[100])
    dist_from_10 = dist_matrix[150, :]
    print("dist_from_10", dist_from_10.shape)
    img = dist_from_10.reshape(cost_map.shape)
    print("img", img.shape)
    plt.imshow(img, vmax=30)
    # maximum value in the img is 300
    plt.show()
