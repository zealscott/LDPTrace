from typing import Tuple, List
import random


class Grid:
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 step_x: float,
                 step_y: float,
                 index: Tuple[int, int]):
        """
        Attributes:
            min_x, min_y, max_x, max_y: boundary of current grid
            index = (i, j): grid index in the matrix
        """
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = min_x + step_x
        self.max_y = min_y + step_y
        self.index = index

    def in_cell(self, p: Tuple[float, float]):
        if self.min_x <= p[0] <= self.max_x and self.min_y <= p[1] <= self.max_y:
            return True
        else:
            return False

    def sample_point(self):
        x = self.min_x + random.random() * (self.max_x - self.min_x)
        y = self.min_y + random.random() * (self.max_y - self.min_y)

        return x, y

    def equal(self, other):
        return self.index == other.index


class GridMap:
    def __init__(self,
                 n: int,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float):
        """
        Geographical map after griding
        Parameters:
             n: cell count
             min_x, min_y, max_x, max_y: boundary of the map
        """
        min_x -= 1e-6
        min_y -= 1e-6
        max_x += 1e-6
        max_y += 1e-6
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        step_x = (max_x - min_x) / n
        step_y = (max_y - min_y) / n
        self.step_x = step_x
        self.step_y = step_y

        # Spatial map, n x n matrix of grids
        self.map: List[List[Grid]] = list()
        for i in range(n):
            self.map.append(list())
            for j in range(n):
                self.map[i].append(Grid(min_x + step_x * i, min_y + step_y * j, step_x, step_y, (i, j)))

    def find_shortest_path(self, start: Grid, end: Grid):
        start_i, start_j = start.index
        end_i, end_j = end.index

        shortest_path = list()
        current_i, current_j = start_i, start_j

        while True:
            # NOTICE: shortest path doesn't include the end grid

            shortest_path.append(self.map[current_i][current_j])
            if end_i > current_i:
                current_i += 1
            elif end_i < current_i:
                current_i -= 1
            if end_j > current_j:
                current_j += 1
            elif end_j < current_j:
                current_j -= 1

            if end_i == current_i and end_j == current_j:
                break

        return shortest_path

    def get_adjacent(self, g: Grid) -> List[Tuple[int, int]]:
        """
        Get 8 adjacent grids of g
        """
        i, j = g.index
        adjacent_index = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
                          (i, j - 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        adjacent_index_new = []
        # Remove grids out of bound
        for index in adjacent_index:
            if len(self.map) > index[0] >= 0 and len(self.map[0]) > index[1] >= 0:
                adjacent_index_new.append(index)
        return adjacent_index_new

    def is_adjacent_grids(self, g1: Grid, g2: Grid):
        return True if g2.index in self.get_adjacent(g1) else False

    def bounding_box(self, g1: Grid, g2: Grid):
        """
        Return all grids in the rectangular bounding box EXCEPT g1 and g2
        """
        start_i = min(g1.index[0], g2.index[0])
        start_j = min(g1.index[1], g2.index[1])
        end_i = max(g1.index[0], g2.index[0])
        end_j = max(g1.index[1], g2.index[1])

        box = []
        for i in range(start_i, end_i + 1):
            for j in range(start_j, end_j + 1):
                g = self.map[i][j]
                if not (g.index == g1.index or g.index == g2.index):
                    box.append(g)

        return box

    def get_list_map(self):
        list_map = []
        for li in self.map:
            list_map.extend(li)
        return list_map

    @property
    def size(self):
        return len(self.map) * len(self.map[0])


def is_adjacent_grids(g1: Grid, g2: Grid):
    """
    Doesn't consider the boundary of the map.
    Only use this function when there's no global grid_map.
    """
    i1, j1 = g1.index
    i2, j2 = g2.index
    # East, Northeast, Southeast
    if i2 == i1 + 1 and (j2 == j1 or j2 == j1 + 1 or j2 == j1 - 1):
        return True
    # West, Northwest, Southwest
    if i2 == i1 - 1 and (j2 == j1 or j2 == j1 + 1 or j2 == j1 - 1):
        return True
    # North, South
    if i2 == i1 and (j2 == j1 + 1 or j2 == j1 - 1):
        return True
    return False

