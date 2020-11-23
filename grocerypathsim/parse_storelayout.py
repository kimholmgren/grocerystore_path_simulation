import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage.feature

class StoreLayout:
    def __init__(self, layout_fp, legend, resize_shape=(500, 500)):
        """
        Initialize store layout object
        :param imagefp: filepath to store layout image
        :param legend: dataframe from parse_legend
        :param resize_shape: size to resize all layouts to
        """
        # read the image
        self.layout_orig = cv2.imread(layout_fp)
        # resize to standard size
        self.resize_shape = resize_shape
        self.layout = cv2.resize(self.layout_orig, resize_shape)
        # store legend
        self.legend = legend
        # get masks
        self.masks = self.get_masks()
        # get perimeter options
        self.product_options = self.get_product_placement_options(self.masks)
        # specifically store the index of checkout
        self.checkout_index = legend[legend['Department']=='checkout'].index[0]
        # get combined department blocks, assume customer can walk anywhere else
        self.walkable = sum([self.masks[i] for i in self.masks.keys()])
        # can walk at 1, can't walk at 0
        self.walkable = (~self.walkable.astype(bool)).astype(int)


    def get_masks(self):
        """
        Get the pixels which are within the color threshold for each department
        :return: dictionary of {dpt_index: pixel_mask}
        """
        masks = {}
        for i in range(len(self.legend)):
            masks[i] = cv2.inRange(self.layout, self.legend.iloc[i]['Low_BGR'],
                                                self.legend.iloc[i]['High_BGR'])
        return masks

    def plot_masks(self, masks):
        """
        Plot all department masks to make sure they look as expected,
        may need to modify fig height and width
        :param masks: Dictionary of department index to pixel mask
        :return: Plot of each department
        """
        plt.clf()
        fig, axs = plt.subplots(1, len(masks), sharey=True)
        fig.set_figheight(25)
        fig.set_figwidth(25)
        for i, k in enumerate(masks.keys()):
            axs[i].imshow(masks[k])
            axs[i].title.set_text(self.legend.iloc[i]['Department'])

    def get_perim(self, mask):
        """
        Get the perimeter from a department color mask by checking if the
        pixel has a different value than the one next to it in the x or y
        direction
        :param mask: Color mask from get_masks
        :return: New mask, which contains only perimeter edges
        """
        im = mask.astype(bool)
        im_edge1 = np.logical_xor(im, np.roll(im, 1, axis=0))
        im_edge2 = np.logical_xor(im, np.roll(im, -1, axis=0))
        im_edge3 = np.logical_xor(im, np.roll(im, 1, axis=1))
        im_edge4 = np.logical_xor(im, np.roll(im, -1, axis=1))
        edges = im_edge1 + im_edge2 + im_edge3 + im_edge4
        return edges

    def plot_perims(self, masks):
        """
        Plot the perimeter of each mask for visual inspection. Very large
        size chosen in order to see edges, may need to be increased with more
        departments
        :return: Plot of perimeters is shown
        """
        plt.clf()
        fig, axs = plt.subplots(1, len(masks), sharey=True)
        fig.set_figheight(105)
        fig.set_figwidth(105)
        for i, k in enumerate(masks.keys()):
            axs[i].imshow(get_perim(masks[k]))
            axs[i].title.set_text(self.legend.iloc[i]['Department'])

    def get_perim_indices(self, edges, tol=10):
        """
        Get pixel indices for a given perimeter
        :param tol: how far a product must be placed from an edge, in pixels
        :return: [xs, ys] for perimeter given edges
        """
        per = np.where(edges)
        within_y_limits = np.logical_and(per[0] > tol, per[0] <
                                         self.resize_shape[0] - tol)
        within_x_limits = np.logical_and(per[1] > tol, per[1] <
                                         self.resize_shape[1] - tol)
        within_limits = np.logical_and(within_x_limits, within_y_limits)
        per = [per[0][within_limits], per[1][within_limits]]
        return np.array(per)

    def get_product_placement_options(self, masks):
        """
        Get the places a product can be placed for each department
        :param masks: masks for each department
        :return: edge options for each department
        """
        edge_options = {}
        for i in range(len(self.legend)):
            edge_options[i] = self.get_perim_indices(self.get_perim(masks[i]))
        return edge_options