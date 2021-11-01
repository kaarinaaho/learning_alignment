#! /opt/anaconda3/envs/align/bin/python

# -*- coding: utf-8 -*-
# Copyright 2021 The align-experiment Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for analysis."""


import numpy as np
import scipy as sp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from os.path import dirname, abspath
import sys

d = dirname(abspath(__file__))
sys.path.append(
    d
    )
sys.path.append(
    dirname(d)
    )

from utils import get_95_ci


def calc_dist(x1, y1, x2, y2):
    """Calculate distance between two points."""
    sqr = (x1 - x2)**2 + (y1 - y2)**2
    return np.sqrt(sqr)


def pairwise_distance(systemA, systemB):
    """
    Calculate pairwise distances between points in two systems.

    Args:
    - systemA and systemB: nxd
    """
    n = systemA.shape[0]
    B_transpose = np.transpose(systemB)

    inner = -2 * np.matmul(systemA, B_transpose)

    A_squares = np.sum(
        np.square(systemA), axis=-1
        )
    A_squares = np.transpose(np.tile(A_squares, (n, 1)))

    B_squares = np.transpose(
        np.sum(np.square(systemB), axis=-1)
        )
    B_squares = np.tile(B_squares, (n, 1))

    pairwise_distances = np.sqrt(
        np.abs(
            inner + A_squares + B_squares
            )
        )

    return pairwise_distances


def alignment_correlation(systemA, systemB):
    """Assumes systems are in the same space."""
    def f(x, y):
        return np.sqrt(np.sum((x-y)**2, axis=1))

    # Index of upper triangular matrices
    idx_upper = np.triu_indices(systemA.shape[0], 1)

    # Pairwise distance matrix between system A and system B
    pairwise_both = pairwise_distance(systemA, systemB)

    # Take upper diagonal of corresponding sim matrices for A->B
    vec_A = f(systemA[idx_upper[0]], systemA[idx_upper[1]])
    vec_B = f(systemB[idx_upper[0]], systemB[idx_upper[1]])

    # Spearman correlation
    r_s = sp.stats.spearmanr(vec_A, vec_B)[0]

    return r_s


def plot_block_means_2conditions(df, col_name="correct", unit="block",
                                 fn=None,
                                 title=None):
    """Plot means of input metric by unit time and alignment condition.

    Arguments:
        -df: dataframe containing data for plot
        -col_name: name of column to be plotted on y-axis
        -unit: name of column to be plotted on x-axis
    """
    color_dict = {
        "aligned": "#2ab7ca",
        "misaligned": "#fe4a49"
    }

    yaxis_dict = {
        "correct": "Proportion correct trials",
        "dist_from_correct": "Distance from correct response",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 7), sharey=True)

    plot_df = df.loc[df["trial_type"] == "trial-mainTrial"]

    by_pid = plot_df.copy()
    plot_df = plot_df[["pid", "align_condition", unit, col_name]].groupby(
            ["align_condition", unit]
        ).agg(
            {col_name: ["mean", "std"],
                "pid": ["nunique"]}
            )
    plot_df.reset_index(inplace=True)
    plot_df.columns = [
        "align_condition", unit, "mean_correct", "std_correct", "n_pid"
        ]
    plot_df = get_95_ci(plot_df)

    # Plot confidence intervals
    for a_c in list(set(plot_df["align_condition"])):
        ac_df = plot_df.loc[plot_df["align_condition"] == a_c]
        ax.fill_between(ac_df[unit],
                        ac_df["upperci"], ac_df["lowerci"],
                        color=color_dict[a_c], alpha=0.3)

    # Plot mean lines
    for a_c in list(set(plot_df["align_condition"])):
        ac_df = plot_df.loc[plot_df["align_condition"] == a_c]
        ax.scatter(ac_df[unit],
                   ac_df["mean_correct"],
                   color=color_dict[a_c],
                   marker="x")
        ax.plot(ac_df[unit],
                ac_df["mean_correct"],
                color=color_dict[a_c])

    ax.set_xlabel(unit.capitalize())
    ax.set_ylabel(yaxis_dict[col_name])
    ax.set_xticks([x for x in range(len(list(set(plot_df[unit]))))],
                  [x for x in range(1, 6)])

    if col_name == "correct":
        ax.set_ylim(-0.05, 1.05)
        chance = 1/6
    if col_name == "dist_from_correct":
        # Calculate chance performance
        a = np.array(
            [[0, 0], [0, 0.6], [0, 1], [0.4, 1], [1, 1], [1, 0.6]])
        pw = pairwise_distance(a, a)
        chance = np.mean(pw)

    ax.set_xticks(
        [
            x for x
            in range(
                int(
                    df.loc[df["trial_type"] == "trial-mainTrial", unit].max()
                    ) + 1
                    )
        ]
    )
    ax.set_xticklabels(
        [
            x + 1 for x
            in range(
                int(
                    df.loc[df["trial_type"] == "trial-mainTrial", unit].max()
                    ) + 1
                    )
        ]
        )
    ax.hlines(chance, 0, 4, color="lightgrey", linestyles="dashed")

    patchList = []
    for key in color_dict:
        data_key = mpatches.Patch(color=color_dict[key], label=key)
        patchList.append(data_key)
    # plt.ylabel("Proportion correct trials")
    plt.legend(handles=patchList)

    if title is not None:
        plt.title(title)

    if fn is not None:
        plt.savefig(fn)
    plt.show()
