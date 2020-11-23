import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class PathGenerator:
    def __init__(self, storelayout, start_coords=[499,0]):
        """
        Initialize path generator object
        :param storelayout: store layout object
        :param start_coords: where customers are generated spatially
        """
        self.slayout = storelayout
        self.dpt_coord_choices = [self.slayout.product_options[i].shape[1] for
                                  i in
                             range(len(self.slayout.product_options.keys()))]
        self.start_coords = start_coords

    def generate_pixel_coordinates(self, shopping_list):
        """
        Generate a random set of coordinates to visit based on the
        departments in a given shopping list
        :param shopping_list: Current shopping list generated from
        ShoppingListGen
        :return: list of pixel coordinates to visit
        """
        visited_dpts = shopping_list['mapped_dpt']
        visited_pixel_coords = []
        for d in visited_dpts:
            curr_pixel_ind_choice = np.random.choice(self.dpt_coord_choices[d])
            curr_pixel = self.slayout.product_options[d][:,
                         curr_pixel_ind_choice]
            visited_pixel_coords.append(curr_pixel)
        visited_pixel_coords = np.array(visited_pixel_coords)
        return visited_pixel_coords

    def order_coords(self, pixel_coords):
        """
        Generate the order for the path from a list of coordinates to visit
        :param pixel_coords: Coordinates to visit
        :return: ordered list of coordinates
        """
        euclidean_dist = 0
        ordered_path = [np.array(self.start_coords)]
        curr_loc = self.start_coords
        while len(pixel_coords)>0:
            # compute euclidean distances from current location
            dists = [np.linalg.norm(a - curr_loc) for a in pixel_coords]
            for i, d in enumerate(dists):
                if d==0:
                    dists[i]=.5
            # compute probabilities
            p = np.power(np.reciprocal(dists), 5)
            p = p / p.sum()
            # choose next point
            next_point_index = np.random.choice(list(range(len(p))), p=p)
            euclidean_dist += dists[next_point_index]
            next_point = pixel_coords[next_point_index]
            pixel_coords = np.vstack((pixel_coords[:next_point_index],
                                      pixel_coords[next_point_index+1:]))
            # add to ordered list
            ordered_path.append(next_point)
            curr_loc = next_point
        # when no items remain visit the checkout area
        checkout_ind = np.random.choice(self.dpt_coord_choices[
                                            self.slayout.checkout_index])
        checkout_point = self.slayout.product_options[
            self.slayout.checkout_index][:, checkout_ind]
        ordered_path.append(checkout_point)
        return ordered_path, euclidean_dist
        # now we have an ordered list of points to visit, and a store layout
        # denoting where we can walk if we choose to compute a path around
        # obstacles rather than euclidean distance

    def calc_path_astar(self, ordered):
        """
        Calculate the walking path using the A* algorithm
        :param ordered: ordered set of coordinates.
        :return: path, distance
        """
        distance = 0
        full_path = []
        # make sure all destination points are walkable
        for o in ordered:
            x, y = o
            self.slayout.walkable[x, y] = 1

        # calculate path
        for i in range(len(ordered)-1):
            # define the grid and the solver
            grid = Grid(matrix=self.slayout.walkable)
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            start = ordered[i]
            start_node = grid.node(start[1], start[0])
            end = ordered[i+1]
            end_node = grid.node(end[1], end[0])
            path, runs = finder.find_path(start_node, end_node, grid)
            distance += len(path)
            full_path.extend(path)
        return full_path, distance


    def plot_ordered_coords(self, visited_pixel_coords):
        plt.imshow(cv2.cvtColor(self.slayout.layout, cv2.COLOR_BGR2RGB))
        xs, ys = [p[0] for p in visited_pixel_coords], [p[1] for p in visited_pixel_coords]
        plt.scatter(ys, xs, color="purple")
        for i in range(len(visited_pixel_coords)):
            plt.text(visited_pixel_coords[i][1] - 10,
                     visited_pixel_coords[i][0] + 25, str(i))
        plt.show()

    def plot_astar_path(self, full_path, ordered):
        plt.imshow(cv2.cvtColor(self.slayout.layout, cv2.COLOR_BGR2RGB))
        xs, ys = [p[0] for p in full_path], [p[1] for p in full_path]
        plt.scatter(xs, ys, color="gray", s=10)
        xs, ys = [p[0] for p in ordered], [p[1] for p in ordered]
        plt.scatter(ys, xs, marker='x', color='red')
        for i in range(len(ordered)):
            plt.text(ordered[i][1] - 10,
                     ordered[i][0] + 25, str(i), fontsize='large',
                     fontdict={'weight': 'heavy', 'color': 'black'})
        plt.show()

    def plot_euclidean_path(self, visited_pixel_coords):
        plt.clf()
        plt.imshow(cv2.cvtColor(self.slayout.layout, cv2.COLOR_BGR2RGB))
        xs, ys = [p[0] for p in visited_pixel_coords], [p[1] for p in
                                                        visited_pixel_coords]

        plt.scatter(ys, xs, marker='x', color='red', zorder=2)
        for i in range(len(visited_pixel_coords)):
            plt.text(visited_pixel_coords[i][1] - 10,
                     visited_pixel_coords[i][0] + 25, str(i), fontsize='large',
                     fontdict={'weight': 'heavy', 'color': 'black'})
        plt.plot(ys, xs, color="gray", linewidth=4, zorder=1)
        plt.show()
