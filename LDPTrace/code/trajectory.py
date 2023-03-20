from typing import List, Tuple
from grid import GridMap, Grid
import grid
import utils
import map_func
import numpy as np


def trajectory_point2grid(t: List[Tuple[float, float]], g: GridMap, interp=True):
    """
    Convert trajectory from raw points to grids
    :param t: raw trajectory
    :param g: grid map
    :param interp: whether to interpolate
    :return: grid trajectory
    """
    grid_map = g.map
    grid_t = list()

    for p in range(len(t)):
        point = t[p]
        found = False
        # Find which grid the point belongs to
        for i in range(len(grid_map)):
            for j in range(len(grid_map[i])):
                if grid_map[i][j].in_cell(point):
                    grid_t.append(grid_map[i][j])
                    found = True
                    break
            if found:
                break

    # Remove duplicates
    grid_t_new = [grid_t[0]]
    for i in range(1, len(grid_t)):
        if not grid_t[i].index == grid_t_new[-1].index:
            grid_t_new.append(grid_t[i])

    # Interpolation
    if interp:
        grid_t_final = list()
        for i in range(len(grid_t_new)-1):
            current_grid = grid_t_new[i]
            next_grid = grid_t_new[i+1]
            # Adjacent, no need to interpolate
            if grid.is_adjacent_grids(current_grid, next_grid):
                grid_t_final.append(current_grid)
            else:
                # Result of find_shortest_path() doesn't include the end point
                grid_t_final.extend(g.find_shortest_path(current_grid, next_grid))

        grid_t_final.append(grid_t_new[-1])
        return grid_t_final

    return grid_t_new


def trajectory_grid2points(g_t: List[Grid]):
    if len(g_t) == 1:
        return [g_t[0].sample_point() for _ in range(2)]
    return [g.sample_point() for g in g_t]


def pass_through(t: List[Grid], g: Grid):
    for t_g in t:
        if t_g.index == g.index:
            return True

    return False


def get_diameter(t: List[Tuple[float, float]]):
    max_d = 0
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            max_d = max(max_d, utils.euclidean_distance(t[i], t[j]))

    return max_d


def get_travel_distance(t: List[Tuple[float, float]]):
    dist = 0
    for i in range(len(t) - 1):
        curr_p = t[i]
        next_p = t[i+1]
        dist += utils.euclidean_distance(curr_p, next_p)

    return dist


def get_real_markov(grid_db: List[List[Grid]], grid_map: GridMap):
    markov_vec = np.zeros(grid_map.size * 8)
    for t in grid_db:
        for i in range(len(t) - 1):
            curr_grid = t[i]
            next_grid = t[i + 1]
            map_id = map_func.adjacent_pair_grid_map_func((curr_grid, next_grid), grid_map)
            markov_vec[map_id] += 1

    return markov_vec
