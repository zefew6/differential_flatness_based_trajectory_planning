"""
A* on 2D gridmap
"""

import queue
import numpy as np


# manhatun_dist
def manhatun_dist(idx1, idx2, res):
    return abs(idx1[0] - idx2[0]) * res + abs(idx1[1] - idx2[1]) * res

def dist(idx1, idx2, res):
    l1 = (idx1[0] - idx2[0]) * res
    l2 = (idx1[1] - idx2[1]) * res
    return np.sqrt(l1 * l1 + l2 * l2)


def getPath(Last_idx, p, gridmap):
    path = []
    count = 0
    cur_idx = Last_idx

    while cur_idx is not None:
        count += 1
        coord = gridmap.index_to_coor(cur_idx)
        path.append(np.asarray(coord))
        cur_idx = p[cur_idx]
    path = np.asarray(path).reshape(count, 2)
    path = path[::-1, :]
    return path


def Astar(start, goal, gridmap):
    g = np.full((gridmap.grid.shape[0], gridmap.grid.shape[1]), np.inf)
    p = np.empty_like(gridmap.grid, dtype=tuple)
    state = np.zeros((gridmap.grid.shape[0], gridmap.grid.shape[1]))

    pq = queue.PriorityQueue()

    start_index = tuple(gridmap.coor_to_index(start))
    goal_index = tuple(gridmap.coor_to_index(goal))

    hasReach = 0
    Last_idx = None
    count = 0

    g[start_index] = 0

    pq.put((manhatun_dist(start_index, goal_index, gridmap.resolution) + 0, start_index))

    while not pq.empty():
        cur = pq.get()
        cur_index = cur[1]
        cur_g = g[cur_index]
        state[cur_index] = -1
        count += 1

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                
                if i == 0 and j == 0:
                    continue

                neighbor_idx = (cur_index[0] + i, cur_index[1] + j)

                if not gridmap.is_valid_index(neighbor_idx):
                    continue

                if gridmap.is_occupied_index(neighbor_idx):
                    continue

                if state[neighbor_idx] == -1:  # not in pq
                    continue

                if neighbor_idx == goal_index:
                    p[goal_index] = cur_index
                    Last_idx = goal_index
                    hasReach = 1
                    break

                d = cur_g + dist(cur_index, neighbor_idx, gridmap.resolution)

                if state[neighbor_idx] == 0:
                    state[neighbor_idx] = 1
                    g[neighbor_idx] = d
                    p[neighbor_idx] = cur_index
                    f = manhatun_dist(neighbor_idx, goal_index, gridmap.resolution) + g[neighbor_idx]
                    pq.put((f, neighbor_idx))

                elif d < g[neighbor_idx]:
                    g[neighbor_idx] = d
                    p[neighbor_idx] = cur_index
                    f = manhatun_dist(neighbor_idx, goal_index, gridmap.resolution) + g[neighbor_idx]
                    pq.put((f, neighbor_idx))

        if hasReach == 1:
            break

    if hasReach == 1:
        path = getPath(Last_idx, p, gridmap)
        # change the coord of start and end point
        path[0] = start
        path[-1] = goal

        return path, count
    else:
        return None, count



def graph_search(start, goal, gridmap):
    path, count = Astar(start, goal, gridmap)
    if path is None:
        print("No path found!")
        return None
    else:
        print(f"Path found with {len(path)} waypoints, searched {count} nodes.")
        return path
