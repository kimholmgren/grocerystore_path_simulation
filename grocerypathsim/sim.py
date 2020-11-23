import datetime as dt
from collections import defaultdict
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def calc_dists(curr_groc, pathgen, astar=True):
    pix_coords = pathgen.generate_pixel_coordinates(curr_groc)
    ordered, euc_dist = pathgen.order_coords(pix_coords)
    if astar:
        full_path, astar_dist = pathgen.calc_path_astar(ordered)
    else:
        astar_dist = None
    return euc_dist, astar_dist

def simulate(pathgens, listgenerator, num_customers=100, astar=True, disp=True):
    results = defaultdict(list)
    if disp:
        print("Start Time", dt.datetime.now())
    prev_time = dt.datetime.now()
    for i in range(num_customers):
        curr_groc = next(listgenerator)
        for j, layout in enumerate(pathgens):
            curr_dists = calc_dists(curr_groc, layout, astar)
            if astar:
                results[str(j+1)+'astar'].append(curr_dists[1])
            results[str(j+1)+'euc'].append(curr_dists[0])
        # track timing
        curr_time = dt.datetime.now()
        took = curr_time - prev_time
        secs = took.days * 24 * 60 * 60 + took.seconds
        results['time (seconds)'].append(secs)
        results['list_length'].append(len(curr_groc))
        prev_time = curr_time
        if disp:
            print("Iteration "+str(i), curr_time, secs)
    res_df = pd.DataFrame.from_dict(results)
    return res_df

def compute_paired_CI(df, col1, col2, alpha=.05):
    D = df[col1] - df[col2]
    #var_p = df[col1].var() + df[col2].var() - 2* df[[col1, col2]].cov(
    # ).iloc[0, 1]
    s_sq_d = 1/(len(df)-1) * np.square(D-D.mean()).sum()
    t_val = stats.t.ppf(1-alpha/2, len(df)-1)
    lo = D.mean() - t_val * np.sqrt(s_sq_d/len(df))
    hi = D.mean() + t_val * np.sqrt(s_sq_d/len(df))
    return lo, D.mean(), hi


def plot_paired_CIs(cis, pairing_names, path_type):
    plt.axhline(color='black')
    sz = len(cis)
    colors = ['orange', 'green', 'blue', 'gray', 'red', 'yellow', 'purple']
    if len(cis)>len(colors):
        colors = [list(np.random.random(size=3))  for i in range(sz)]
    else:
        colors = colors[:len(cis)]
    xs = list(range(sz))
    plt.scatter(xs, [ci[0] for ci in cis], marker=6, c=colors)
    plt.scatter(xs, [ci[2] for ci in cis], marker=7, c=colors)
    plt.scatter(xs, [ci[1] for ci in cis], marker='.', c=colors)
    for i in range(sz):
        plt.plot([i]*3, cis[i], c=colors[i], label="Confidence Interval of Layout "+pairing_names[i])
    plt.title("Paired Confidence Intervals for Each Layout Combination, "+path_type)
    plt.xticks(xs, pairing_names)
    plt.legend()
    plt.show()