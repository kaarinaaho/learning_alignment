#! /opt/anaconda3/envs/align/bin/python
from os.path import join, dirname, abspath
import sys
import torch
import torch.nn.functional as F
import torch.distributions as D

import numpy as np

import pandas as pd
import random

d = dirname(abspath(__file__))
sys.path.append(
     d
    )

def get_trials(positions, n_blocks=5, unsup_per_block=2):

    trials = []
    trial_types = []
    meta_block_length = unsup_per_block + 1

    for i in range(meta_block_length * n_blocks):

        shuff_blocks = random.sample([x for x in range(len(positions))], len(positions))
        trials = trials + shuff_blocks

        if i % meta_block_length == 2:
            trial_types = trial_types + (["supervised"] * len(positions))
        else:
            trial_types = trial_types + (["unsupervised"] * len(positions))

    return trials, trial_types


def get_95_ci(df):
    """Add columns for upper and lower bounds of 95% CI to input df."""
    print(df)
    df['upperci'] = df.apply(
        lambda x: (
            x["mean_correct"] + 1.96 * x["std_correct"]/np.sqrt(x["n_pid"])
            ), axis=1
        )
    df["lowerci"] = df.apply(
        lambda x: (
            x["mean_correct"] - 1.96 * x["std_correct"]/np.sqrt(x["n_pid"])
            ), axis=1
        )

    return df


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
        torch.unsqueeze(probs, 1).cpu().detach().numpy() * space.cpu().detach().numpy(),
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
        """
        if(any(np.isnan(probs.detach().numpy()))):
            print(probs.detach().numpy())
            #print(np.array([(np.sqrt(2) - x.detach().numpy())/temp for x in dists]))
        """
        if torch.isnan(probs).any() | torch.isinf(probs).any():
            print(probs)
        idx_select = torch.multinomial(probs, 1)

    return idx_select


def get_all_dists(x, y):

    return torch.sqrt(torch.sum(torch.pow(x - y, 2), 1))


def branched(probs, alpha):
    """Branch from probability dist to random with prob 1-alpha."""
    probs = torch.mul(probs, alpha)

    # Add to random
    random = torch.mul(torch.ones(probs.shape[0]), 1/probs.shape[0])
    probs = torch.add(probs, torch.mul(random, (1-alpha)))

    return probs


def get_pid_map(pid):
    maps = pd.read_csv(join(
                            dirname(dirname(dirname(d))),
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
    events = pd.read_csv(join(
                            dirname(dirname(dirname(d))),
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
    maps = pd.read_csv(join(
                            dirname(dirname(dirname(d))),
                            "data/experiment/raw/all_maps.csv"
                            ), header=None, sep=";")
    maps.columns = ["pid", "houseID", "house_x", "house_y", "creature_x",
                    "creature_y", "trial_id", "trial_type", "position_type"]
    maps = maps.drop_duplicates()

    gen_info = {}
    pid_map = maps.loc[(maps["pid"] == pid) & (maps["position_type"] == "trial") & (maps["trial_type"] == "generalisation")]

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
