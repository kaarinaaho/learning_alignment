#! /opt/anaconda3/envs/align/bin/python
from os.path import join, dirname, abspath
import sys
import torch
import torch.nn.functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import pandas as pd
import random

d = dirname(abspath(__file__))
sys.path.append(
     d
    )


def generate_trials(positions, n_blocks=5, unsup_per_block=2):
    """Generate sequence of trials in structure of expt."""
    trials = []
    trial_types = []
    meta_block_length = unsup_per_block + 1

    for i in range(meta_block_length * n_blocks):

        shuff_blocks = random.sample(
            [x for x in range(len(positions))], len(positions)
            )
        trials = trials + shuff_blocks

        if i % meta_block_length == 2:
            trial_types = trial_types + (["supervised"] * len(positions))
        else:
            trial_types = trial_types + (["unsupervised"] * len(positions))

    return trials, trial_types


def _euclidean_distance(x, y):
    """L2 distance."""
    return torch.mean(torch.sqrt(torch.sum(torch.pow(x - y, 2), 1)))


def cycle_loss_func(x, x_c, loss_cycle_scale):
    """Cycle loss."""
    loss = loss_cycle_scale * (
        _euclidean_distance(x, x_c)
        )
    return loss


def set_loss_func(fx, gmm_y_samples, gmm_scale, device):
    """Statistical distance between two sets.

    Use upperbound of GMM KL divergence approximation.
    Assumes that incoming `fx` and `y` is all of the data.
    """
    n_concept = fx.shape[0]
    n_dim = fx.shape[1]

    # Equal weight for each component
    mixture = D.Categorical(torch.ones(n_concept,).to(device))
    # Diagonal covariance matrix scaled by gmm_scale
    components = D.Independent(
        D.Normal(fx, gmm_scale * torch.ones(n_dim,).to(device)), 1
        )
    gmm_fx = D.mixture_same_family.MixtureSameFamily(mixture, components)

    # Negative log-likelihood
    neg_ll = - torch.mean(gmm_fx.log_prob(gmm_y_samples), axis=0)
    return neg_ll


def get_weighted_pos(probs, space):
    """Get expected position of item based on prob dist across houses."""
    return np.sum(
        torch.unsqueeze(
            probs, 1
        ).cpu().detach().numpy() * space.cpu().detach().numpy(),
        axis=0)


def dists_to_probs(dists, device, temp=1):
    """Extract probability of choices from distances to choices.

    - dists: distances on which to form distribution
    - temp: temperature parameter, controlling confidence of prob dist
      (temperature values above 1 will flatten the probability dist,
      while values below one will accentuate it)
    """
    # Subtract distance from max possible distance in space; incorporate temp
    roots = torch.sqrt(torch.multiply(torch.ones(dists.size()), 2)).to(device)
    dists = torch.add(roots, - torch.div(dists, temp))

    # Max norm
    dists = torch.subtract(dists, torch.max(dists))

    probs = F.softmax(dists, dim=0)

    if torch.isnan(probs).any() | torch.isinf(probs).any():
        print(probs, dists)

    return probs


def choose_from_probs(probs, dists, policy="greedy"):
    """Select option based on probability distribution across choices.

    policy = "greedy": selects the option with highest probability on
    each trial
    policy = "probabilistic": random sample from the probability
    distribution across options
    """
    if policy == "greedy":
        idx_select = torch.argmax(probs)

    elif policy == "probabilistic":
        if torch.isnan(probs).any() | torch.isinf(probs).any():
            print(probs)
        idx_select = torch.multinomial(probs, 1)

    return idx_select


def get_all_dists(x, y):
    """Get L2 distance of x from each item in y."""
    return torch.sqrt(torch.sum(torch.pow(x - y, 2), 1))


def branched(probs, alpha):
    """Branch from probability dist to random with prob 1-alpha."""
    probs = torch.mul(probs, alpha)

    # Add to random
    random = torch.mul(torch.ones(probs.shape[0]), 1/probs.shape[0])
    probs = torch.add(probs, torch.mul(random, (1-alpha)))

    return probs


def get_pid_map(pid):
    """Get pid's assigned map in expt from raw data."""
    maps = pd.read_csv(join(
                            dirname(dirname(d)),
                            "data/experiment/raw/all_maps.csv"
                            ), header=None, sep=";")
    maps.columns = ["pid", "houseID", "house_x", "house_y", "creature_x",
                    "creature_y", "trial_id", "trial_type", "position_type"]
    maps = maps.drop_duplicates()

    # Get participant's map for supervised trials
    pid_map = maps.loc[
        maps["pid"] == pid
        ].loc[
            ((maps["position_type"] == "trial")
             & (maps["trial_type"] == "main"))
            ]
    pid_map = pid_map.drop(
        columns=["position_type", "trial_type", "houseID"])

    pid_map = pid_map.astype(
        {'house_x': 'float',
         'house_y': 'float',
         'creature_x': 'float',
         'creature_y': 'float'}
        )

    # Compile map into np arrays accepted by model classes
    pid_map["spaceY"] = [
        [int(pid_map["house_x"].iloc[i]), int(pid_map["house_y"].iloc[i])]
        for i in range(len(pid_map["house_x"]))
        ]
    pid_map["spaceX"] = [
        [
            int(pid_map["creature_x"].iloc[i]),
            int(pid_map["creature_y"].iloc[i])
        ]
        for i in range(len(pid_map["creature_x"]))]

    pid_map["trial_id"] = pid_map["trial_id"].rank() - 1
    pid_map = pid_map.sort_values("trial_id")
    if all(pid_map["spaceX"] == pid_map["spaceY"]):
        pid_map["align_condition"] = "aligned"
    else:
        pid_map["align_condition"] = "misaligned"

    return pid_map


def get_pid_trials(pid):
    """Get pid's assigned series of trials in expt from raw data."""
    events = pd.read_csv(join(
                            dirname(dirname(d)),
                            "data/experiment/raw/all_events.csv"
                            ), sep=";", header=None)
    events.columns = [
        "pid", "event_type", "t", "trial_no", "trial_id", "event_detail"
        ]

    # Filter for participant and for trial page loads
    pid_events = events.loc[events["pid"] == pid]
    pid_events = pid_events.loc[
        (pid_events["event_type"] == "unsupervised") |
        (
            (pid_events["event_type"] == "load")
            & (pid_events["event_detail"] == "trial-mainTrial")
        )]

    # Get ranked unsupervised trials
    unsup = pid_events.loc[(pid_events["event_type"] == "unsupervised")]
    unsup.loc[:, "trial_no"] = unsup.loc[:, "t"].rank() - 1

    # Dedupe supervised trial loads
    sup = pid_events[
        ((pid_events["event_type"] == "load")
         & (pid_events["event_detail"] == "trial-mainTrial"))]

    pid_events = pd.concat([sup, unsup])
    pid_events = pid_events.drop_duplicates(
        subset=["event_type", "event_detail", "trial_no", "trial_id"]
        )
    pid_events["trial_no"] = pid_events["t"].rank() - 1

    # Co-ordinate trial IDs for observational trials
    pid_events.loc[
        pid_events["event_type"] == "unsupervised", "trial_id"
        ] = pid_events.loc[
            pid_events["event_type"] == "unsupervised", "event_detail"
            ]

    # Align naming for supervised/choice trials
    pid_events.loc[
        pid_events["event_type"] != "unsupervised", "event_type"
        ] = "supervised"
    pid_events["trial_id"] = pid_events["trial_id"].astype(int)
    pid_events["trial_id"] = pid_events["trial_id"].rank(method="dense") - 1

    pid_events = pid_events.sort_values("trial_no")

    return pid_events


def get_pid_generalisation(pid):
    """Get pid's assigned generalisation trial from raw expt data."""
    maps = pd.read_csv(join(
                            dirname(dirname(d)),
                            "data/experiment/raw/all_maps.csv"
                            ), header=None, sep=";")
    maps.columns = ["pid", "houseID", "house_x", "house_y", "creature_x",
                    "creature_y", "trial_id", "trial_type", "position_type"]
    maps = maps.drop_duplicates()

    gen_info = {}
    pid_map = maps.loc[(
                         (maps["pid"] == pid)
                         & (maps["position_type"] == "trial")
                         & (maps["trial_type"] == "generalisation")
                        )]

    gen_info["x_gen"] = np.array([
        pd.to_numeric(
            pid_map.loc[pid_map["houseID"] == "zeroshot-correct", "creature_x"]
            ).iloc[0],
        pd.to_numeric(
            pid_map.loc[pid_map["houseID"] == "zeroshot-correct", "creature_y"]
            ).iloc[0]
        ])

    gen_info["y_gen_options"] = [
        np.array([
            pd.to_numeric(
             pid_map.loc[pid_map["houseID"] == "zeroshot-correct", "house_x"]
                ).iloc[0],
            pd.to_numeric(
             pid_map.loc[pid_map["houseID"] == "zeroshot-correct", "house_y"]
                ).iloc[0]
        ]),
        np.array([
            pd.to_numeric(
             pid_map.loc[pid_map["houseID"] == "zeroshot-incorrect", "house_x"]
                ).iloc[0],
            pd.to_numeric(
             pid_map.loc[pid_map["houseID"] == "zeroshot-incorrect", "house_y"]
                ).iloc[0]
        ])
    ]
    gen_info["y_gen_idxcorrect"] = 0

    return gen_info


def append_random_model(df):
    """Add random model to candidate models for best fit."""
    # Append random model
    random_df = df[["pid", "align_condition"]].drop_duplicates()
    random_df["model"] = "random"
    random_df[
                [
                 "lr_sup", "lr_unsup", "lr", "lam_a_cyc", "lam_dist",
                 "hidden_size", "s", "params"
                ]
            ] = 0
    random_df["loss"] = - 30 * np.log(1/6)
    df = pd.concat([df, random_df])
    return(df)


def calculate_AIC(df):
    """Calculate Akaike's Information Criterion."""
    # Implement AIC 2k - 2ln(L)
    df["params"] = 3
    df.loc[df["model"] == "cycle_and_distribution", "params"] = 5
    df.loc[df["model"] == "random", "params"] = 0
    df["AIC"] = 2*df["params"] + 2 * df["loss"]
    df["ranked_AIC"] = df.groupby("pid")["AIC"].rank("dense", ascending=True)

    return df


def plot_best_fits(best_fits, plt_type="count", save=False):
    """Plot barplots to count which model is best fit."""
    models_included = append_random_model(best_fits)
    models_included = calculate_AIC(models_included)

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
            ((
                np.float32(x) - np.float32(sel["AIC"].iloc[i])
             )/np.float32(x)) * 100
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
