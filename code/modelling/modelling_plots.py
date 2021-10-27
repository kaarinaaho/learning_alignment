#! /opt/anaconda3/envs/align/bin/python

import pandas as pd
import numpy as np

import modelling_utils as mu
import processing.process_utils as utils

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import pandas as pd


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
    plot_df = mu.get_95_ci(plot_df)

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
        pw = utils.pairwise_distance(a, a)
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


def plot_best_fits(best_fits, plt_type="count", save=False):

    models_included = mu.append_random_model(best_fits)
    models_included = mu.calculate_AIC(models_included)

    colors = ["#2ab7ca", "#fe4a49"]
    sns.set_palette(sns.color_palette(colors))

    sel = models_included.loc[models_included["ranked_AIC"] == 1]

    plt_params = {
        "count": {
            "y": ("pid", "nunique"),
            "y_lab": "Count best fit model",
            "x_labs": [
                "Regression + Aligner", "Classifier", "Regression", "Random"
                ],
            "fn": "Barplot_modelfits.png"
        },
        "AIC_uplift": {
            "y": "baseline_uplift",
            "y_lab": "% AIC improvement from random",
            "x_labs": ["Regression + Aligner", "Classifier", "Regression"],
            "fn": "Barplot_AICimprovement.png"
        }
    }

    if plt_type == "count":
        sel = sel[
                    ["model", "pid", "align_condition", "AIC", "loss"]
                 ].groupby(["model", "align_condition"]).agg(
            {
                "AIC": ["mean", "std"],
                "loss": ["mean", "std"],
                "pid": ["nunique"]
            }).reset_index()

    if plt_type == "AIC_uplift":
        sel["aic_baseline"] = 107.51
        sel["baseline_uplift"] = [
            ((np.float32(x) - np.float32(sel["AIC"].iloc[i]))/np.float32(x))*100
            for i, x in enumerate(sel["aic_baseline"])
            ]
        sel = sel.loc[sel["model"] != "random"]

    plt.figure()
    sns.barplot(
        x="model", y=plt_params[plt_type]["y"],
        hue="align_condition", data=sel
        )
    plt.ylabel(plt_params[plt_type]["y_lab"])
    plt.xlabel("")
    plt.xticks(
        ticks=[i for i in range(len(plt_params[plt_type]["x_labs"]))],
        labels=plt_params[plt_type]["x_labs"]
        )
    plt.legend(title="")
    if save is not False:
        plt.savefig(plt_params[plt_type]["fn"])
    plt.show()
