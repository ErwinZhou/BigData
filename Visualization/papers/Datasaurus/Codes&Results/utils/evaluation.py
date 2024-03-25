#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Usage:
    evaluation.py run [<relative_path>]
    evaluation.py -h | --help
    
    Based on the code in same_stats.py,
    Same Stats, Different Graphs Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing
    This is an evaluation script for model's performance on
    approximating the target trajectory.
    The script will calculate the Frechet distance and Hausdorff distance
    between the target trajectory and the approximated trajectory.

    Version1.0
    Now the evaluation script in only based on given hard-coded target trajectory such as a circle.
    And it has been extended to be cover any interval between sampling data during the transformation in the model.
    To be continue......

"""
import numpy as np
import pandas
from docopt import docopt
from frechetdist import frdist
from scipy.spatial.distance import directed_hausdorff
import math
from matplotlib import pyplot as plt


def get_points_on_circle(cx, cy, r, num_points):
    """
    This function will generate points on a circle based on the hard-coded center and radius in the paper.
    :param cx: center x of the circle
    :param cy: center y of the circle
    :param r: radius of the circle
    :param num_points: points of circle to be generated and sampled, it should be identical to the number of the
    approximated trajectory to guarantee a same dimension for distance calculation.
    :return:
    """
    circle_points = []
    for i in range(num_points):
        angle = math.pi * 2 * i / num_points
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        circle_points.append((x, y))
    return circle_points





def plot_evaluation(frechet_dists, hausdorff_dists):
    """
    This function will plot the evaluation result of the approximated trajectory
    Based on the Frechet distance and Hausdorff distance.
    :param frechet_dists: Calculated Frechet distance from do_evaluation
    :param hausdorff_dists: Calculated Hausdorff distance from do_evaluation
    :return:
    """
    iterations = [i * 50000 for i in range(7)]
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, frechet_dists, marker='o', color='b', label='Frechet Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Frechet Distance')
    plt.title('Frechet Distance of Approximated Trajectory')
    plt.xticks(iterations)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, hausdorff_dists, marker='v', color='r', label='Hausdorff Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Hausdorff Distance')
    plt.title('Hausdorff Distance of Approximated Trajectory')
    plt.xticks(iterations)
    plt.grid(True)
    plt.legend()
    plt.show()


def do_evaluation(file_path):
    """
    This function will calculate the Frechet distance and Hausdorff distance
    to give a rough evaluation of how approximated the model can be.
    :param file_path: given parameter of the file path, prepared to be loaded
    :return:
    """
    circle_points = get_points_on_circle(54.26, 47.83, 30, 142)
    target_trajectory = circle_points
    frechet_dists = []
    hausdorff_dists = []
    for i in range(7):
        csv_path = file_path + "/circle-data-0000" + str(i) + ".csv"
        approximated_trajectory = pandas.read_csv(csv_path, usecols=[1, 2]).values.tolist()
        approximated_trajectory = np.array(approximated_trajectory)
        target_trajectory = np.array(target_trajectory)
        print("-----------------Evaluation-----------------")
        print("Iteration: ", 50000 * i)
        # print("Target trajectory: ", target_trajectory)
        # print("Approximated trajectory: ", approximated_trajectory)
        print("-----Frechet Distance-----")
        frechet_dist = frdist(target_trajectory, approximated_trajectory)
        print("The Frechet distance is:" + str(frechet_dist))
        frechet_dists.append(frechet_dist)
        print("-----Hausdorff Distance-----")
        hausdorff_dist = directed_hausdorff(target_trajectory, approximated_trajectory)[0]
        print("The Hausdorff distance is:" + str(hausdorff_dist))
        hausdorff_dists.append(hausdorff_dist)
    # Call the plot function at the end of do_evaluation
    plot_evaluation(frechet_dists, hausdorff_dists)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Evaluation 1.0')
    file_path = None
    if arguments['run']:
        if arguments['<relative_path>']:
            file_path = "../results/" + arguments['<relative_path>']
            do_evaluation(file_path)
        else:
            print("No relative path")
