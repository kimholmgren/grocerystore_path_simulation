import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def get_department_labels(img):
    """
    Use Tesseract OCR and split on newlines to get department names from a
    store legend. Manual repair may be necessary, this usually isn't perfect
    :param img: legend image read in with opencv
    :return: List of department names, lowercased and stripped of special
    characters
    """
    txt = pytesseract.image_to_string(img, config='--psm 6')
    dpts = txt.split('\n')
    dpts = [''.join(c for c in d if  c not in '|_?:!/;[]()').strip() for d in dpts]
    #dpts = [d.strip('| ').strip(')').strip('|').strip() for d in dpts]
    dpts = [d.lower() for d in dpts if d!='']
    return dpts

def is_grayscale(pixel, thresh=20):
    """
    We make the assumption that gray, black, and white are not used as colors in
    the legend. If this assumption is violated manual repair be necessary.
    This is done in order to separate legend colors from background and
    outline colors, which may be black/white/gray.
    :param pixel: 3x1 array, BGR pixel color value
    :param thresh: int, how far away all pixels are from each other if the
    pixel is considered gray
    :return: True/False - whether the pixel is grayscale
    """
    if np.abs(pixel[0]-pixel[1])<thresh and np.abs(pixel[1]-pixel[2])<thresh and np.abs(pixel[0]-pixel[2])<thresh:
        return True
    else:
        return False

def get_centroids(legend, dpts, min_pixel_pct=.0005):
    """
    Find the unique non-grayscale images in a legend, filter by those that
    appear over min_pixel_pct percent of the image, use K Means to cluster
    into len(dpts) unique, common colors. The centroids of those means should
    be the legend colors
    :param legend: legend image read in with opencv
    :param dpts: list of departments from OCR, used to determine how many
    colors to identify
    :param min_pixel_pct: The minimum percent of the pixels that must be a
    color for it to be considered in the clustering. Used to remove smaller
    variations in color around the border. Defaults to .05% of the image.
    :return: List of len(dpts) BGR centroids
    """
    min_pixels = legend.shape[0] * legend.shape[1] * min_pixel_pct
    all_rgbs = legend.reshape(-1, legend.shape[-1])
    # get unique rgbs
    unique_rgbs, rgb_ct = np.unique(all_rgbs, axis=0, return_counts=True)
    # remove grayscale (includes white background and black text/outline)
    gray = np.array([is_grayscale(x.astype(int)) for x in unique_rgbs])
    rgbs, rgb_ct = unique_rgbs[~gray], rgb_ct[~gray]
    # only include those that appear over min times
    unique_rgbs_overmin = rgbs[rgb_ct>min_pixels]
    rgb_ct_overmin = rgb_ct[rgb_ct>min_pixels]
    # fit a k means classifier
    clf = KMeans(n_clusters=len(dpts))
    clf.fit(unique_rgbs_overmin)
    centroids = clf.cluster_centers_.astype(int)
    return centroids

def create_centroid_ranges(centroids, tol=15):
    """
    Create high and low BGR values to allow for slight variations in layout
    :param centroids: legend colors identified from get_centroids
    :param tol: int, number of pixels any of B, G, R could vary by and still
    be considered part of the original centroid. Defaults to 15.
    :return: List of [low, high] bounds for each centroid
    """
    cent_bounds = []
    for c in centroids:
        b, g, r = c
        cent_bounds.append(
            [[b - tol, g - tol, r - tol], [b + tol, g + tol, r + tol]])
    return cent_bounds

def create_colormap_df(legend, dpts):
    """
    Get centroids, map centroids to their vertical positions,
    which correspond to the order of departments. Create dataframe containing
    department name, BGR centroid, high and low BGR, and the RGB value
    :param legend: legend image, opened by cv2
    :param dpts: department names, identified by get_department_labels and
    manually repaired if necessary
    :return: dataframe mapping department to color
    """
    # get centroids
    centroids = get_centroids(legend, dpts)
    cent_bounds = create_centroid_ranges(centroids)
    # get average height for each color center
    height = []
    for i in range(len(centroids)):
        pxls = np.where(cv2.inRange(legend, np.array(cent_bounds[i][0]),
                                    np.array(cent_bounds[i][1])))
        height.append(pxls[0].mean())
    # sort to put in dataframe
    sorted_colors = centroids[np.argsort(height)]
    low_bounds = np.array(cent_bounds)[:, 0][np.argsort(height)]
    high_bounds = np.array(cent_bounds)[:, 1][np.argsort(height)]
    df = pd.DataFrame.from_dict(
        {"Department": dpts, "BGR_Centroid": list(sorted_colors),
         "Low_BGR": list(low_bounds), "High_BGR": list(high_bounds),
         "RGB": list(np.flip(sorted_colors, axis=1))})
    return df

def plot_colormap(df):
    """
    Plot the colormap in the dataframe to visually verify assignment of
    colors to departments
    :param df: Dataframe of legend colormap created in create_colormap_df
    :return: show plot of colors and department names
    """
    r = len(df)
    plt.scatter([1] * r, [r - i * 1 for i in range(r)], c=list(df['RGB'] /
                                                               255), s=200,
                edgecolors='black')
    dpts = list(df['Department'])
    for i, d in enumerate(dpts):
        plt.text(1.001, r - i, d)
    plt.show()

