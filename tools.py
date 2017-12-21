# coding: utf-8

import math
import numpy as np
import pandas as pd

from time import time
import matplotlib.pyplot as plt
from geopy.distance import vincenty

from scipy.stats import randint

from sklearn import linear_model, ensemble, svm
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, LeaveOneOut, \
    LeaveOneGroupOut, StratifiedShuffleSplit, train_test_split


def vincenty_vec(vec_coord):
    vin_vec_dist = np.zeros(vec_coord.shape[0])
    if vec_coord.shape[1] !=  4:
        print('ERROR: Bad number of columns (shall be = 4)')
    else:
        vin_vec_dist = [vincenty(vec_coord[m ,0:2] ,vec_coord[m ,2:]).meters for m in range(vec_coord.shape[0])]
    return vin_vec_dist


def Eval_geoloc(y_train_lat , y_train_lng, y_pred_lat, y_pred_lng):
    """
    Evaluate distance error for each predicted point

    :param y_train_lat:
    :param y_train_lng:
    :param y_pred_lat:
    :param y_pred_lng:
    :return:
    """
    vec_coord = np.array([y_train_lat , y_train_lng, y_pred_lat, y_pred_lng])
    err_vec = vincenty_vec(np.transpose(vec_coord))
    return err_vec


def plot_error_distribution(err_vec):
    """
    Plot error distribution + error criterion

    :param err_vec:
    :return:
    """
    # Plot error distribution
    values, base = np.histogram(err_vec, bins=50000)
    cumulative = np.cumsum(values)
    plt.figure();
    plt.plot(base[:-1 ] /1000, cumulative / np.float(np.sum(values))  * 100.0, c='blue')
    plt.grid(); plt.xlabel('Distance Error (km)'); plt.ylabel('Cum proba (%)'); plt.axis([0, 30, 0, 100]);
    plt.title('Error Cumulative Probability'); plt.legend( ["Opt LLR", "LLR 95", "LLR 99"])


def compute_and_plot_error_distribution(y_train_lat , y_train_lng, y_pred_lat, y_pred_lng):
    """
    Compute and plot error distribution.

    :param y_train_lat:
    :param y_train_lng:
    :param y_pred_lat:
    :param y_pred_lng:
    :return:
    """
    err_vec = Eval_geoloc(y_train_lat, y_train_lng, y_pred_lat, y_pred_lng)

    # Error criterion
    error_criterion = np.percentile(err_vec, 80)
    print("Error criterion : ", error_criterion)

    plot_error_distribution(err_vec)
    return err_vec
