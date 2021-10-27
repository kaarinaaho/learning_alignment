#! /opt/anaconda3/envs/align/bin/python

from os.path import dirname, abspath
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import NLLLoss
import torch.optim as optim


import numpy as np

import pandas as pd
from hyperopt import hp

import modelling_utils as mu

from itertools import compress
import random

d = dirname(abspath(__file__))
sys.path.append(
        d
    )


class model_expt:

    def __init__(self, align_condition, spaceX, spaceY=None,
                 trials=None, trial_types=None, n_blocks=5,
                 unsup_per_block=2, generalise=None):
        """Initialize."""
        self.device = torch.device("cpu")

        # Normalise space along each axis
        spaceX = (spaceX-np.min(spaceX, axis=0))
        spaceX = spaceX/np.max(spaceX, axis=0)

        self.x = torch.FloatTensor(spaceX).to(self.device)
        self.align_condition = align_condition

        if spaceY is None:
            if align_condition == "aligned":
                self.y = torch.FloatTensor(spaceX).to(self.device)
            elif align_condition == "misaligned":
                self.y = torch.FloatTensor(
                    random.sample(list(spaceX), len(spaceX))).to(self.device)
        else:
            spaceY = (spaceY-np.min(spaceY, axis=0))
            spaceY = spaceY/np.max(spaceY, axis=0)
            self.y = torch.FloatTensor(spaceY).to(self.device)

        if trials is None:
            trials, trial_types = mu.get_trials(
                spaceX, n_blocks, unsup_per_block
                )
            self.trials = trials
            self.trial_types = trial_types

        else:
            self.trials = trials
            self.trial_types = trial_types

        if generalise is None:
            self.generalise = False

        else:
            self.generalise = True
            self.x_gen = torch.FloatTensor(generalise["x_gen"])
            self.y_gen_options = torch.FloatTensor(generalise["y_gen_options"])
            self.y_gen_idxcorrect = generalise["y_gen_idxcorrect"]

    def train_euclidean(self, hidden_size=100, lr_sup=0.1, lr_unsup=0.1,
                        steps_per_trial=30, temp=1, alpha=1,
                        get_prob_dists=False, eps=1e-30):

        model_f = MLP(2, hidden_size, 2).to(self.device)

        params = list(model_f.parameters())
        optimizer_sup = optim.SGD(params, lr_sup)
        optimizer_unsup = optim.SGD(params, lr_unsup)

        trial_correct_resp = []
        loss_ls = []
        supervision_labels = []
        prob_dists = []
        weighted_pos_list = []
        flag = False

        for i, trial in enumerate(self.trials):

            x_trial = self.x[trial]

            # Distances of output to each option
            y_c = model_f(x_trial)
            trial_correct_resp.append(trial)
            dists = mu.get_all_dists(torch.unsqueeze(y_c, 0), self.y)

            # Probability distribution across options
            probs = mu.dists_to_probs(dists, self.device, temp=temp)

            # Add epsilon
            probs = torch.add(probs, eps)
            probs = torch.div(probs, torch.sum(probs))

            # Implement branching
            probs = mu.branched(probs, alpha)

            prob_dists.append(
                [i]
                + [self.trial_types[i][0]] + list(probs.cpu().detach().numpy())
                )

            # Weighted pred position
            weighted_pos = mu.get_weighted_pos(probs, self.y)
            weighted_pos_list.append(weighted_pos)

            supervision_labels.append(self.trial_types[i][0])

            # Update model
            for s in range(steps_per_trial):
                optimizer_sup.zero_grad()
                optimizer_unsup.zero_grad()
                y_c = model_f(x_trial)
                if (y_c.isnan().any()):
                    # Break for exploding grad in hyperparam testing
                    flag = True
                    break
                dists = mu.get_all_dists(torch.unsqueeze(y_c, 0), self.y)

                # Get cross-entropy loss
                probs = mu.dists_to_probs(dists, self.device, temp=temp)

                # Add epsilon
                probs = torch.add(probs, eps)
                probs = torch.div(probs, torch.sum(probs))

                # Implement branching
                probs = mu.branched(probs, alpha)
                probs = torch.log(probs)

                ce_loss = NLLLoss()
                loss = ce_loss(
                    torch.unsqueeze(probs, 0),
                    torch.tensor([trial]).to(self.device)
                    )

                loss.backward()
                if self.trial_types[i] == "supervised":
                    optimizer_sup.step()
                else:
                    optimizer_unsup.step()

            if flag:
                break

            loss_ls.append(loss.cpu().detach().numpy())

        if flag:
            cols = ["correct_response", "correct_x", "correct_y",
                    "sup_CE_loss", "trial_no", "block", "supervised",
                    "weighted_pos_x", "weighted_pos_y"]
            df = pd.DataFrame(dict(zip(
                cols, [None] * len(cols)
            )), index=[0])

        else:
            df = pd.DataFrame({
                "correct_response": trial_correct_resp,
                "correct_x": [
                    self.y[i][0].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "correct_y": [
                    self.y[i][1].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "sup_CE_loss": loss_ls,
                "trial_no": [i for i in range(len(loss_ls))],
                "block": [
                    np.floor(i/self.x.size()[0])
                    for i in range(len(loss_ls))
                    ],
                "supervised": supervision_labels,
                "weighted_pos_x": [x[0] for x in weighted_pos_list],
                "weighted_pos_y": [x[1] for x in weighted_pos_list]
            })

        if self.generalise:
            y_gen = model_f(self.x_gen)
            dists = mu.get_all_dists(torch.unsqueeze(y_gen, 0), self.y_gen_options)

            # Get cross-entropy loss
            probs_gen = mu.dists_to_probs(dists, self.device, temp=temp)

            # Add epsilon
            probs_gen = torch.add(probs_gen, eps)
            probs_gen = torch.div(probs_gen, torch.sum(probs_gen))

            # Implement branching
            probs_gen = mu.branched(probs_gen, alpha)

            # Prob correct
            prob_gen = probs_gen.cpu().detach().numpy()[self.y_gen_idxcorrect]
            df["prob_generalisation"] = prob_gen

        if get_prob_dists:
            prob_dists = pd.DataFrame(
                prob_dists,
                columns=["trial_no", "supervised", 0, 1, 2, 3, 4, 5]
                )
            return df, prob_dists
        else:
            return df

    def train_classifier(self, hidden_size=100, lr_sup=0.1, lr_unsup=0.1,
                         steps_per_trial=30, temp=1, alpha=1,
                         get_prob_dists=False, eps=1e-30):

        model_f = MLP_classify(
                2, hidden_size, self.x.size()[0]
            ).to(self.device)

        params = list(model_f.parameters())
        optimizer_sup = optim.SGD(params, lr_sup)
        optimizer_unsup = optim.SGD(params, lr_unsup)

        trial_correct_resp = []
        loss_ls = []
        supervision_labels = []
        prob_dists = []
        weighted_pos_list = []
        flag = False

        for i, trial in enumerate(self.trials):

            x_trial = self.x[trial]

            # Guess before update
            out = model_f(x_trial)
            w_temp = torch.div(out, temp)
            probs = F.softmax(torch.subtract(w_temp, torch.max(w_temp)), dim=0)

            # Add epsilon
            probs = torch.add(probs, eps)
            probs = torch.div(probs, torch.sum(probs))

            prob_dists.append(
                [i]
                + [self.trial_types[i][0]]
                + list(probs.cpu().detach().numpy()))

            trial_correct_resp.append(trial)
            supervision_labels.append(self.trial_types[i][0])

            # Weighted pred position
            weighted_pos = mu.get_weighted_pos(probs, self.y)
            weighted_pos_list.append(weighted_pos)

            # Update
            for s in range(steps_per_trial):
                optimizer_sup.zero_grad()
                optimizer_unsup.zero_grad()

                out = model_f(x_trial)
                if (out.isnan().any()):
                    # Break for exploding grad in hyperparam testing
                    flag = True
                    break
                w_temp = torch.div(out, temp)
                probs = F.softmax(
                    torch.subtract(w_temp, torch.max(w_temp)), dim=0
                    )

                # Add epsilon
                probs = torch.add(probs, eps)
                probs = torch.div(probs, torch.sum(probs))

                # NLL requires logprob input
                probs = torch.log(probs)

                ce_loss = NLLLoss()
                loss = ce_loss(
                    torch.unsqueeze(probs, 0),
                    torch.tensor([trial]).to(self.device)
                    )

                loss.backward()

                if self.trial_types[i] == "supervised":
                    optimizer_sup.step()
                else:
                    optimizer_unsup.step()

            if flag:
                break

            loss_ls.append(loss.cpu().detach().numpy())

        if flag:
            cols = ["correct_response", "correct_x", "correct_y",
                    "sup_CE_loss", "trial_no", "block", "supervised",
                    "weighted_pos_x", "weighted_pos_y"]
            df = pd.DataFrame(dict(zip(
                cols, [None] * len(cols)
            )), index=[0])

        else:
            df = pd.DataFrame({
                "correct_response": trial_correct_resp,
                "correct_x": [
                    self.y[i][0].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "correct_y": [
                    self.y[i][1].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "sup_CE_loss": loss_ls,
                "trial_no": [i for i in range(len(loss_ls))],
                "block": [
                    np.floor(i/self.x.size()[0])
                    for i in range(len(loss_ls))
                    ],
                "supervised": supervision_labels,
                "weighted_pos_x": [x[0] for x in weighted_pos_list],
                "weighted_pos_y": [x[1] for x in weighted_pos_list]
            })

        if self.generalise:
            # Generalisation is random guess
            df["prob_generalisation"] = 0.5

        if get_prob_dists:
            prob_dists = pd.DataFrame(
                prob_dists,
                columns=["trial_no", "supervised", 0, 1, 2, 3, 4, 5]
                )
            return df, prob_dists
        else:
            return df

    def train_cycle(self, hidden_size=10, lr_sup=0.1, lr_unsup=0.1,
                    lam_a_cyc=1, lam_sup=1, lam_s_cyc=1, lam_dist=0,
                    steps_per_trial=30, temp=1, alpha=1, get_prob_dists=False,
                    eps=1e-30):

        model_f = MLP(2, hidden_size, 2).to(self.device)
        model_g = MLP(2, hidden_size, 2).to(self.device)

        params = list(model_f.parameters()) + list(model_g.parameters())
        optimizer_sup = optim.SGD(params, lr_sup)
        optimizer_unsup = optim.SGD(params, lr_unsup)

        trial_correct_resp = []
        a_cyc_loss_ls = []
        sup_loss_ls = []
        dist_loss_ls = []
        supervision_labels = []
        prob_dists = []
        weighted_pos_list = []

        flag = False

        for i, trial in enumerate(self.trials):

            x_trial = torch.unsqueeze(self.x[trial], 0)
            y_trial = torch.unsqueeze(self.y[trial], 0)

            # Distances of output to each option
            y_c = model_f(x_trial)
            trial_correct_resp.append(trial)
            dists = mu.get_all_dists(y_c, self.y)

            # Probability distribution across options
            probs = mu.dists_to_probs(dists, self.device, temp=temp)

            # Add epsilon
            probs = torch.add(probs, eps)
            probs = torch.div(probs, torch.sum(probs))

            # Implement branching
            probs = mu.branched(probs, alpha)

            prob_dists.append(
                [i]
                + [self.trial_types[i][0]]
                + list(probs.cpu().detach().numpy())
                )

            # Track trial type
            supervision_labels.append(self.trial_types[i][0])

            # Weighted pred position
            weighted_pos = mu.get_weighted_pos(probs, self.y)
            weighted_pos_list.append(weighted_pos)

            # Update
            for s in range(steps_per_trial):
                optimizer_sup.zero_grad()
                optimizer_unsup.zero_grad()

                # Supervised component
                f_x = model_f(x_trial)
                g_y = model_g(y_trial)

                if (f_x.isnan().any()) | (g_y.isnan().any()):
                    # Break for exploding grad in hyperparam testing
                    flag = True
                    break

                dists_y = mu.get_all_dists(f_x, self.y)
                dists_x = mu.get_all_dists(g_y, self.x)

                # Get cross-entropy loss
                probs_x = mu.dists_to_probs(dists_x, self.device, temp=temp)
                probs_y = mu.dists_to_probs(dists_y, self.device, temp=temp)

                # Add epsilon
                probs_x = torch.add(probs_x, eps)
                probs_x = torch.div(probs_x, torch.sum(probs_x))

                # Implement branching
                probs_x = mu.branched(probs_x, alpha)
                probs_x = torch.log(probs_x)

                probs_y = torch.add(probs_y, eps)
                probs_y = torch.div(probs_y, torch.sum(probs_y))

                # Implement branching
                probs_y = mu.branched(probs_y, alpha)
                probs_y = torch.log(probs_y)

                ce_loss_x = NLLLoss()
                ce_loss_y = NLLLoss()
                loss_x = ce_loss_x(
                    torch.unsqueeze(probs_x, 0),
                    torch.tensor([trial]).to(self.device)
                    )
                loss_y = ce_loss_y(
                    torch.unsqueeze(probs_y, 0),
                    torch.tensor([trial]).to(self.device)
                    )
                sup_loss = lam_sup * 0.5 * (loss_x + loss_y)
                loss = sup_loss

                a_cyc_loss = torch.tensor(0).to(self.device)
                dist_loss = torch.tensor(0).to(self.device)

                # After first passive block, add full_batch_cycle_loss
                if i > 5:

                    optimizer_sup.zero_grad()
                    optimizer_unsup.zero_grad()
                    # Cycle - batch size = all

                    # Shuffle items to ensure no cheating
                    shuff_1 = np.random.permutation([x for x in range(6)])
                    g_f_all_x = model_g(model_f(self.x[shuff_1]))
                    a_cyc_loss_y = lam_a_cyc * mu._euclidean_distance(
                        g_f_all_x, self.x[shuff_1]
                        )

                    shuff_2 = np.random.permutation([x for x in range(6)])
                    f_g_all_y = model_f(model_g(self.y[shuff_2]))
                    a_cyc_loss_x = lam_a_cyc * mu._euclidean_distance(
                        f_g_all_y, self.y[shuff_2]
                        )

                    a_cyc_loss = 0.5 * (a_cyc_loss_x + a_cyc_loss_y)

                    # Dist component (batch_size = all)
                    shuff_3 = np.random.permutation([x for x in range(6)])
                    f_all_x = model_f(self.x[shuff_3])
                    dist_loss_y = lam_dist * mu.set_loss_func(
                        f_all_x, self.y, 0.1, self.device
                        )

                    shuff_4 = np.random.permutation([x for x in range(6)])
                    g_all_y = model_g(self.y[shuff_4])
                    dist_loss_x = lam_dist * mu.set_loss_func(
                        g_all_y, self.x, 0.1, self.device
                        )

                    dist_loss = dist_loss_x + dist_loss_y

                    batch_loss = dist_loss + a_cyc_loss

                    loss = loss + batch_loss

                loss.backward()
                if self.trial_types[i] == "supervised":
                    optimizer_sup.step()
                else:
                    optimizer_unsup.step()

            if flag:
                break

            a_cyc_loss_ls.append(a_cyc_loss.cpu().detach().numpy())
            sup_loss_ls.append(sup_loss.cpu().detach().numpy())
            dist_loss_ls.append(dist_loss.cpu().detach().numpy())

        if flag:
            cols = ["correct_response", "correct_x", "correct_y",
                    "a_cyc_loss", "sup_CE_loss", "dist_loss",
                    "trial_no", "block", "supervised", "weighted_pos_x",
                    "weighted_pos_y"]
            df = pd.DataFrame(dict(zip(
                cols, [None] * len(cols)
            )), index=[0])

        else:
            df = pd.DataFrame({
                "correct_response": trial_correct_resp,
                "correct_x": [
                    self.y[i][0].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "correct_y": [
                    self.y[i][1].cpu().detach().numpy()
                    for i in trial_correct_resp
                    ],
                "a_cyc_loss": a_cyc_loss_ls,
                "sup_CE_loss": sup_loss_ls,
                "dist_loss": dist_loss_ls,
                "trial_no": [i for i in range(len(sup_loss_ls))],
                "block": [
                    np.floor(i/self.x.size()[0])
                    for i in range(len(sup_loss_ls))
                    ],
                "supervised": supervision_labels,
                "weighted_pos_x": [x[0] for x in weighted_pos_list],
                "weighted_pos_y": [x[1] for x in weighted_pos_list]
            })

        if self.generalise:
            y_gen = model_f(self.x_gen)
            dists = mu.get_all_dists(
                torch.unsqueeze(y_gen, 0), self.y_gen_options
                )

            # Get cross-entropy loss
            probs_gen = mu.dists_to_probs(dists, self.device, temp=temp)

            # Add epsilon
            probs_gen = torch.add(probs_gen, eps)
            probs_gen = torch.div(probs_gen, torch.sum(probs_gen))

            # Implement branching
            probs_gen = mu.branched(probs_gen, alpha)

            # Prob correct
            prob_gen = probs_gen.cpu().detach().numpy()[self.y_gen_idxcorrect]
            df["prob_generalisation"] = prob_gen

        if get_prob_dists:
            prob_dists = pd.DataFrame(
                prob_dists,
                columns=["trial_no", "supervised", 0, 1, 2, 3, 4, 5]
                )
            return df, prob_dists
        else:
            return df


class MLP(nn.Module):
    """Simple MLP to map from one space to another.

    Parameters:
        n_dim_in: dimensionality of input
        hidden: number of nodes in hidden layers
        n_dim_out: dimensionality of output

    """

    def __init__(self, n_dim_in, hidden, n_dim_out):
        super(MLP, self).__init__()

        hidden = int(hidden)

        self.fc1 = nn.Linear(n_dim_in, hidden)
        self.fc2 = nn.Linear(hidden, n_dim_out)

        # Initialise with Glorot uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.r = nn.ReLU()

    def forward(self, x):
        """Feed-forward pass."""
        h1 = self.r(self.fc1(x))
        y = torch.sigmoid(self.fc2(h1))

        return y


class MLP_classify(nn.Module):
    """Simple MLP to map from one space to another.

    Parameters:
        n_dim_in: dimensionality of input
        hidden: number of nodes in hidden layers
        n_dim_out: dimensionality of output
    """

    def __init__(self, n_dim_in, hidden, n_classes):
        super(MLP_classify, self).__init__()

        self.fc1 = nn.Linear(n_dim_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

        # Initialise with Glorot uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.r = nn.ReLU()

    def forward(self, x):
        """Feed-forward pass."""
        h1 = self.r(self.fc1(x))
        y = self.fc2(h1)

        return y


def get_hyperparam_min_func(mod_type, align_condition, n_simulations=1, trial_inputs=None,
                            fit_type="MSE_blockmean", q=None, exclude=False):
    """Return function which is inputted for hyperparam optimisation."""

    def fit_mse(params, align_condition=align_condition,
                n_simulations=n_simulations, mod_type=mod_type,
                trial_inputs=trial_inputs, fit_type=fit_type, q=q,
                exclude=exclude):
        """Get function inputted for hyperparam optimisation."""
        # Train model with input parameters and model type
        collect_df, collect_probs = run_expt_flex(
            params,
            align_condition=align_condition,
            n_simulations=n_simulations,
            model_type=mod_type,
            trial_inputs=trial_inputs)

        if len(collect_df["trial_no"]) < n_simulations * 30:
            # Overwrite score if training terminated, e.g  exploding gradients
            score = np.nan

        else:

            sup_filt = [x == "s" for x in collect_probs["supervised"]]
            # Filter probabilities for choice trials
            collect_probs = collect_probs.loc[
                sup_filt
                ]

            # Re-index trials to match with behavioural trials
            collect_probs["trial_no"] = (
                collect_probs["trial_no"].rank(method="dense") - 1
            )

            # Get participant's submitted response on each trial
            submitted_inputs = trial_inputs["submitted_inputs"]

            # Merge probs with behavioural trials
            submitted_inputs = submitted_inputs.merge(
                collect_probs,
                on="trial_no",
                how="right")
            
            #print(submitted_inputs)

            # Minimise NLL of participant responses
            if fit_type == "sumloglik":

                # For each trial, get probability of submitted response
                submitted_inputs["prob"] = [
                    submitted_inputs.loc[
                        i, int(submitted_inputs.loc[i, "sub_idx"])
                        ]
                    for i in range(len(submitted_inputs["sub_idx"]))
                    ]

                # NLL with epsilon for computational reasons, for each t
                submitted_inputs["neglogprob"] = [
                    -np.log(x) for x in submitted_inputs["prob"]
                    ]

                # Score to minimise is sum neg log likelihood
                score = submitted_inputs[
                                        ["pid", "neglogprob"]
                                    ].groupby("pid").agg("sum")
                score = min(score["neglogprob"])

            # Minimise NLL of participant 'correctness'
            elif fit_type == "binaryCE":

                # For each trial, get prob of the correct input
                correct_inputs = pd.DataFrame(
                    zip(
                        list(compress(sup_filt, trial_inputs["trials"])),
                        [x for x in range(len(collect_probs))]
                    ), columns=["correct", "trial_no"]
                )
                submitted_inputs = submitted_inputs.merge(
                    correct_inputs, on="trial_no"
                    )

                # Get boolean column indicating if participant was correct
                submitted_inputs["correct_behaviour"] = [
                    submitted_inputs["correct"].iloc[i] == x
                    for i, x in enumerate(submitted_inputs["sub_idx"])
                    ]

                # For each trial, get p of correct and incorrect responses
                submitted_inputs[True] = [
                    submitted_inputs.loc[
                        i, int(submitted_inputs.loc[i, "correct"])
                        ]
                    for i in range(len(submitted_inputs["correct"]))
                    ]
                submitted_inputs[False] = [1-x for x in submitted_inputs[1]]

                # Select prob for the participant's actual accuracy
                submitted_inputs["prob"] = [
                    submitted_inputs.loc[i, x]
                    for i, x 
                    in enumerate(submitted_inputs["correct_behaviour"])
                ]

                # NLL with epsilon for computational reasons, for each t
                submitted_inputs["neglogprob"] = [
                    -np.log(x) for x in submitted_inputs["prob"]
                    ]

                # Score to minimise is sum neg log likelihood
                score = submitted_inputs[
                                    ["pid", "neglogprob"]
                                        ].groupby("pid").agg("sum")
                score = min(score["neglogprob"])

            if q is not None:
                score = np.round(score/q) * q

        return score

    return fit_mse


def get_param_spaces(model_type):
    """Define parameter search spaces for each model's parameters."""
    space = {
                "lr_pow": hp.quniform('lr_pow', -3, -0.5, 0.3),
                "temp": hp.quniform('temp', 0.2, 1.5, 0.1),
                "alpha": hp.uniform("alpha", 0, 1)
            }

    if (model_type == "cycle_and_distribution"):
        space["lam_a_cyc"] = hp.uniform('lam_a_cyc', 0.6, 2)

    if (model_type == "cycle_and_distribution"):
        space["lam_dist"] = hp.uniform('lam_dist', 0.001, 0.01)

    return space


def run_expt_flex(params, align_condition, trial_inputs,
                  n_simulations=1, model_type="classifier",
                  generalise=None):
    """Take input params and train model given inputted trial details."""
    for p in range(n_simulations):
        if trial_inputs is not None:
            creature_space = trial_inputs["spaceX"]
            model = model_expt(
                align_condition,
                spaceX=trial_inputs["spaceX"],
                spaceY=trial_inputs["spaceY"],
                trials=trial_inputs["trials"],
                trial_types=trial_inputs["trial_types"],
                generalise=generalise)
        else:
            creature_space = np.array(
                [[1, 1], [1, 4], [1, 6], [3, 6], [6, 6], [6, 4]]
                    )
            model = model_expt(
                align_condition,
                spaceX=creature_space,
                generalise=generalise)

        lr_sup = 10 ** params.get("lr_pow")
        lr_unsup = 10 ** params.get("lr_pow")

        s = params.get("s")
        if s is None:
            s = 30

        hidden_size = params.get("hidden_size")
        if hidden_size is None:
            hidden_size = 100

        temp = params.get("temp")

        alpha = params.get("alpha")

        if model_type == "classifier":
            data, probs = model.train_classifier(
                hidden_size=hidden_size,
                lr_sup=lr_sup,
                lr_unsup=lr_unsup,
                steps_per_trial=s,
                get_prob_dists=True,
                temp=temp,
                alpha=alpha
                )

        elif model_type == "euclidean":
            data, probs = model.train_euclidean(
                hidden_size=hidden_size,
                lr_sup=lr_sup,
                lr_unsup=lr_unsup,
                steps_per_trial=s,
                get_prob_dists=True,
                temp=temp,
                alpha=alpha
                )

        elif model_type == "cycle_and_distribution":
            lam_dist = params.get("lam_dist")
            lam_a_cyc = params.get("lam_a_cyc")

            data, probs = model.train_cycle(
                hidden_size=hidden_size,
                lr_sup=lr_sup,
                lr_unsup=lr_unsup,
                lam_a_cyc=lam_a_cyc,
                lam_dist=lam_dist,
                lam_sup=1,
                steps_per_trial=s,
                get_prob_dists=True,
                temp=temp,
                alpha=alpha)

        data["pid"] = p
        data["align_condition"] = align_condition
        data["trial_type"] = "trial-mainTrial"
        probs["pid"] = p

        if p == 0:
            collect_df = data
            collect_probs = probs
        else:
            collect_df = pd.concat([collect_df, data])
            collect_probs = pd.concat([collect_probs, probs])

    return collect_df, collect_probs
