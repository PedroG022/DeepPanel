import numpy as np
from collections import Counter

class Panel:
    def __init__(self):
        self.number_in_page = None
        
        self.left = None
        self.bottom = None
        self.right = None
        self.top = None

        self.name = None

    def __str__(self) -> str:
        return f"""
            Number in page: {self.number_in_page}
            Left: {round(self.left)}
            Width: {round(self.width)}
            Top: {round(self.top)}
            Height: {round(self.height)}
        """

class DeepPanelResult:
    def __init__(self):
        self.connected_component_result: ConnectedComponentResult = None
        self.panels = []

class ConnectedComponentResult:
    def __init__(self, total_clusters, clusters_matrix, pixels_per_labels):
        self.total_clusters = total_clusters
        self.clusters_matrix = clusters_matrix
        self.pixels_per_labels = pixels_per_labels

def uf_find(x):
    y = x
    while labels[y] != y:
        y = labels[y]
    while labels[x] != x:
        z = labels[x]
        labels[x] = y
        x = z
    return y

def uf_union(x, y):
    labels[uf_find(x)] = uf_find(y)
    return labels[uf_find(x)]

def uf_make_set():
    labels[0] += 1
    assert labels[0] < n_labels
    labels[labels[0]] = labels[0]
    return labels[0]

def uf_initialize(max_labels):
    global n_labels, labels
    n_labels = max_labels
    labels = [0] * n_labels

def uf_done():
    global n_labels, labels
    n_labels = 0
    labels = []

def find_components_x(matrix):
    m, n = len(matrix), len(matrix[0])
    uf_initialize(m * n // 2)

    for j in range(n):
        for i in range(m):
            if matrix[i][j]:
                up = matrix[i - 1][j] if i > 0 else 0
                left = matrix[i][j - 1] if j > 0 else 0

                if not up and not left:
                    matrix[i][j] = uf_make_set()
                elif up and not left:
                    matrix[i][j] = up
                elif not up and left:
                    matrix[i][j] = left
                elif up and left:
                    matrix[i][j] = uf_union(up, left)

    new_labels = [0] * n_labels
    pixels_per_label = [0] * n_labels
    for j in range(n):
        for i in range(m):
            if matrix[i][j]:
                x = uf_find(matrix[i][j])
                if new_labels[x] == 0:
                    new_labels[0] += 1
                    new_labels[x] = new_labels[0]
                new_label_to_assign = new_labels[x]
                matrix[i][j] = new_label_to_assign
                pixels_per_label[new_label_to_assign] += 1

    total_clusters = new_labels[0]
    uf_done()


    result = {}
    result['total_clusters'] = total_clusters
    result['clusters_matrix'] = matrix
    result['pixels_per_labels'] = pixels_per_label


    return ConnectedComponentResult(total_clusters, matrix, pixels_per_label)

