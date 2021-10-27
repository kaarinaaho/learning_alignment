#! /opt/anaconda3/envs/align/bin/python

from os.path import dirname, abspath, join
import sys

import process_data as process
import modelling_plots as plot
import pandas as pd

import modelling_main as mm
import modelling_utils as mu
import modelling_plots as plot

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, Trials

import os.path

d = dirname(abspath(__file__))
sys.path.append(
    d
    )

def get_pids(i_d, tot_split=5):
    data = pd.read_csv(join(dirname(dirname(dirname(d))), "data/experiment/processed/Processed_data_filtered.csv"))
    data = data["pid"].drop_duplicates()
    n_pids = len(data)
    interval = int(np.floor(n_pids/tot_split))
    if i_d != tot_split-1:
        pids = list(data.iloc[interval*i_d: interval*(i_d+1)])
    else:
        pids = list(data.iloc[interval*i_d:])
    return pids


def fit_participant_data(
    pids,
    fp="data/models/new_fits/hyperparam_opt_results.csv"
    ):

    # For each participant, read in series of trials and the condition they were assigned to:
    data = pd.read_csv(join(dirname(dirname(dirname(d))), "data/experiment/processed/Processed_data_filtered.csv"))
    data = process.get_submitted_coords(data)

    model_types = ["euclidean", "classifier", "cycle_and_distribution"]

    fp = join(dirname(dirname(dirname(d))), fp)

    if os.path.isfile(fp):
        df = pd.read_csv(
                fp,
                header=0, index_col=0)
    else:
        df = pd.DataFrame({})

    for pid in pids:
        # Load map and event data for participant
        pm = mu.get_pid_map(pid)
        pe = mu.get_pid_trials(pid)

        # Get list of submitted response indexes by trial no
        pid_dat = data.loc[
            (data["pid"] == pid) & (data["trial_type"] == "trial-mainTrial")
            ]
        pid_dat["sub_idx"] = [
            int(list(pm["spaceY"]).index(x)) for x in pid_dat["sub_coords"]
            ]
        pid_dat = pid_dat.sort_values("trial_no")

        # Dict of inputs to experiment class
        trial_inputs = {
            "trials": [int(x) for x in pe["trial_id"]],
            "trial_types": list(pe["event_type"]),
            "submitted_inputs": pid_dat[["trial_no", "sub_idx"]],
            "spaceX": np.array(list(pm["spaceX"])),
            "spaceY": np.array(list(pm["spaceY"])),
        }
        ac = pm["align_condition"].iloc[0]

        # For each model type being tested
        for mod in model_types:

            if len(df) > 0:
                dat = df.loc[(df["model"] == mod) & (df["pid"] == pid)]
            else:
                dat = []

            # If model/participant pair not alredy in df
            if len(dat) == 0:
                if mod == "classifier":
                    n_simulations = 3  # Increased to maximise flexibility
                    max_evals = 200
                else:
                    n_simulations = 1
                    max_evals = 200

                print("Running pid ", pid)

                # Fit model parameters with fit mse
                min_func = mm.get_hyperparam_min_func(
                    mod, ac, trial_inputs=trial_inputs,
                    fit_type="sumloglik", q=None, n_simulations=n_simulations
                )
                space = mm.get_param_spaces(mod)
                trials = Trials()

                # Fit appropriate hyperparameters
                best = fmin(
                        min_func,
                        space=space,
                        algo=tpe.suggest,
                        trials=trials,
                        max_evals=max_evals
                )

                # Extract best loss
                best_loss = min([x["result"]["loss"] for x in trials.trials])

                lr = 10**best.get("lr_pow")

                # Record in a dataframe
                append = pd.DataFrame({
                    "pid": [pid],
                    "model": [mod],
                    "align_condition": [ac],
                    "lr_sup": [best.get("lr_sup")],
                    "lr_unsup": [best.get("lr_unsup")],
                    "lr": [lr],
                    "lam_a_cyc": [best.get("lam_a_cyc")],
                    "lam_dist": [best.get("lam_dist")],
                    "hidden_size": 100,
                    "temp": [best.get("temp")],
                    "s": 30,
                    "loss": best_loss})

                df = pd.concat([df, append])

            # Save best to df of pid, model type, parameters selected and AIC of the resulting fitted model
            df.to_csv(fp)


if __name__ == "__main__":

    pids = get_pids(0, 50)  # Select the first 1/50th of participants
    fit_participant_data(
        pids,
        fp="data/models/new_fits/hyperparam_opt_results.csv"
        )

    best_fits_test = pd.read_csv(
        join(
            dirname(dirname(dirname(d))),
            "data/models/new_fits/hyperparam_opt_results.csv"
        )
    )
    plot.plot_best_fits(best_fits_test, plt_type="count")
    plot.plot_best_fits(best_fits_test, plt_type="AIC_uplift")

    """
    # Hyperparam count plotting functions for full set of fits
    best_fits_all = pd.read_csv(
        join(
            dirname(dirname(dirname(d))),
            "data/models/paper_results/fitted_model_hyperparams.csv"
        )
    )
    plot.plot_best_fits(best_fits_all, plt_type="count")
    plot.plot_best_fits(best_fits_all, plt_type="AIC_uplift")

    # Test
    """
