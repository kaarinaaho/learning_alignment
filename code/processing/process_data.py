#! /opt/anaconda3/envs/align/bin/python

# -*- coding: utf-8 -*-
# Copyright 2021 The learning_alignment Authors. All Rights Reserved.
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
# =============================================================================

"""Processing functions for data collected from align experiment."""
from os.path import join, dirname, abspath
import sys

d = dirname(abspath(__file__))
sys.path.append(
    d
    )
import pandas as pd
import numpy as np
import process_utils as utils
from scipy.stats.mstats import mquantiles


def get_response_entropy(df_in, entropy="by_block", exclude_final=True):
    df = df_in.copy()
    df = df.loc[df["trial_type"] == "trial-mainTrial"]

    if exclude_final is True:
        max_block = df[["pid", "block"]].groupby("pid").agg("max").reset_index()
        max_block.columns = ["pid", "max_block"]
        df = df.merge(max_block, on="pid")
        df = df.loc[df["block"] < df["max_block"]]

    # Get blank values for each house in each block for each pid
    # This is to have 0 next to houses that were not clicked on each
    house_list = list(set(df["submitted_house"]))
    all_houses = pd.concat(
        [df[["pid", "block"]].drop_duplicates()]*len(house_list))
    n_reps = int(len(list(all_houses["pid"]))/len(house_list))
    all_houses["submitted_house"] = list(np.repeat(house_list,n_reps))
    all_houses["trial_no"] = 0

    if entropy=="by_block":
        # Count n times each house is chosen per block
        df = df[["pid", "trial_no", "submitted_house", "block"]].groupby(["pid", "submitted_house", "block"]).agg("count")
        df.reset_index(inplace=True)
        df = pd.concat([df, all_houses])
        df = df.groupby(["pid", "block", "submitted_house"]).agg("sum")
        df.reset_index(inplace=True)
        ts = df[["pid", "block"]].groupby("pid", as_index=False).agg("min")

        df["house_p"] = df["trial_no"]/6
        df["log_p"] = [0 if x == 0 else np.log(x) for x in df["house_p"]]
        df = df.groupby(["pid", "block"]).apply(lambda x: -sum(x["house_p"] * x["log_p"]))
        df = df.reset_index()
        df.columns = ["pid", "block", "entropy"]
        df = df[["pid", "entropy"]].groupby("pid").agg("mean").reset_index()
        df = df_in.merge(df, on=["pid"], how="left")
    
    elif entropy=="overall":
        # Count n times each house is chosen per block
        df = df[["pid", "trial_no", "submitted_house"]].groupby(["pid", "submitted_house"]).agg("count")
        df.reset_index(inplace=True)
        df = pd.concat([df, all_houses])
        df = df.groupby(["pid", "submitted_house"]).agg("sum")
        df.reset_index(inplace=True)

        df["house_p"] = df["trial_no"]/60
        df["log_p"] = [0 if x == 0 else np.log(x) for x in df["house_p"]]
        df = df.groupby(["pid"]).apply(lambda x: -sum(x["house_p"] * x["log_p"]))
        df = df.reset_index()
        df.columns = ["pid", "entropy"]
        df = df_in.merge(df, on=["pid"], how="left")
    return df


def get_all_trials(df):
    """Select all trial events from event log."""
    # Select all trial events (from first trial load to last trial submit)
    events = ["load"]
    details = ["trial-mainTrial", "info-generalisation"]
    relevant_times = df.loc[
        df["event_type"].isin(events) & df["event_detail"].isin(details)
        ]["t"]
    min_trial_t = min(relevant_times)
    max_trial_t = max(relevant_times)
    df_filt = df.loc[df["t"] < max_trial_t]
    df_filt = df_filt.loc[df_filt["t"] >= min_trial_t]

    # Select events from generalisation trial (from load to submit)
    events = ["load"]
    details = ["trial-generalisation", "info-done"]
    relevant_times = df.loc[
        df["event_type"].isin(events) & df["event_detail"].isin(details)
        ]["t"]

    if len(relevant_times) > 0:
        min_trial_t = min(relevant_times)
        max_trial_t = max(relevant_times)
        df_filt2 = df.loc[df["t"] < max_trial_t]
        df_filt2 = df_filt2.loc[df_filt2["t"] >= min_trial_t]
        df_filt = df_filt.append(df_filt2)
    return df_filt


def distance_from_correct(df):
    """Add column for distance between correct and submitted points."""
    df["dist_from_correct"] = df.apply(
        lambda x: utils.calc_dist(
            x['correct_x_normed'],
            x['correct_y_normed'],
            x['submitted_x_normed'],
            x['submitted_y_normed']
            ), axis=1)
    return df


def participant_wise_alignment(df, stimuli="trial"):
    """Get alignment correlation of mapping assigned to participant."""
    if stimuli == "trial":
        box_xs = list(df.loc[df["position_type"] == "trial", "house_x"])
        box_ys = list(df.loc[df["position_type"] == "trial", "house_y"])
        creature_xs = list(
            df.loc[df["position_type"] == "trial", "creature_x"]
            )
        creature_ys = list(
            df.loc[df["position_type"] == "trial", "creature_y"]
            )
    elif stimuli == "all":
        box_xs = list(df.loc[~df["position_type"].isna(), "house_x"])
        box_ys = list(df.loc[~df["position_type"].isna(), "house_y"])
        creature_xs = list(df.loc[~df["position_type"].isna(), "creature_x"])
        creature_ys = list(df.loc[~df["position_type"].isna(), "creature_y"])

    box_coords = np.zeros((len(box_xs), 2))
    creature_coords = np.zeros((len(box_xs), 2))

    # Create np arrays of coordinates for box and creature positions
    for i in range(len(box_xs)):
        box_coords[i][0] = box_xs[i]
        box_coords[i][1] = box_ys[i]
        creature_coords[i][0] = creature_xs[i]
        creature_coords[i][1] = creature_ys[i]

    corr = utils.alignment_correlation(box_coords, creature_coords)
    
    return corr


def print_conditions(data):
    # Filter for status_code > 0
    conditions = data[
            ["align_condition", "rotate_condition", "pid"]
            ].groupby(
                ["align_condition", "rotate_condition"]
                ).agg(
                    "nunique"
                    )

    print(conditions)


def apply_filters(data, entropy=False):

    # Remove anomalous age
    data = data.loc[data["age"]<90]

    print("Full AMT N without 98 year old: ", len(set(data["pid"])))
    print_conditions(data)

    if entropy is True:
        # Filter out bottom 10% response entropy by block
        data = data.loc[(data["entropy"] >= mquantiles(data["entropy"], prob=[0.1])[0])]
        print("Full AMT N filtered for response entropy: ", len(set(data["pid"])))
    
    print_conditions(data)

    return(data)


def get_submitted_coords(data):
    data["sub_coords"] = [
            [x, data["submitted_y"].iloc[i]]
            for i, x in enumerate(data["submitted_x"])
        ]
    return data


def process_infopage_data(
    fp_assignment="/Users/apple/projects/align-experiment/data/\
        all_assignments.csv",
    fp_events="/Users/apple/projects/align-experiment/data/all_events.csv"
        ):
    """Get stats on infopage interactions."""
    # Import assignment data for merge with trial data
    assignment = pd.read_csv(fp_assignment, sep=";", header=None)
    assignment.columns = [
        "pid", "age", "gender", "condition_id", "align_condition",
        "rotate_condition", "t_start", "browser", "platform", "screenHeight",
        "screenWidth", "status_code", "app_version", "x_axis",
        "creature_presence", "colour_0", "orientation_0"
    ]

    # Import events data
    df = pd.read_csv(fp_events, sep=";", header=None)
    df.columns = [
        "pid", "event_type", "t", "trial_no", "trial_id", "event_detail"
        ]
    df = df.drop_duplicates()

    # Info page data
    # Calculate how long they spent on each intro page
    get_pi = df.copy()
    events = ["load", "next"]
    get_pagelengths = get_pi.loc[
        (get_pi['event_type'].isin(events)) &
        (get_pi['event_detail'].str.contains("info"))
        ]
    get_pagelengths = get_pagelengths.pivot(
        values="t", index=["pid", "event_detail"], columns="event_type"
        )
    get_pagelengths["duration"] = (
            get_pagelengths["next"] - get_pagelengths["load"]
        )
    get_pagelengths.reset_index(inplace=True)
    get_pagelengths = get_pagelengths.pivot(
        values="duration", index=["pid"], columns="event_detail"
        )
    infopage_behaviour = get_pagelengths

    # Calculate how many times they pressed the false button
    get_falsepress = get_pi.loc[get_pi['event_type'] == "fake"]
    get_falsepress = get_falsepress[["pid", "event_type"]].groupby(
                            ["pid"]
                        ).agg("count")
    get_falsepress = get_falsepress.rename(
                        columns={"event_type": "fakebutton_press"}
                    )
    infopage_behaviour = infopage_behaviour.merge(get_falsepress, on="pid")

    infopage_behaviour = infopage_behaviour.merge(assignment, on="pid")
    return infopage_behaviour


def process_raw_data(
    fp_assignment=join(
                        dirname(dirname(dirname(d))),
                        "data/experiment/raw/all_assignments.csv"
                       ),
    fp_events=join(
                    dirname(dirname(dirname(d))),
                    "data/experiment/raw/all_events.csv"
                    ),
    fp_map=join(
                dirname(dirname(dirname(d))),
                "data/experiment/raw/all_maps.csv"
                ),
    return_dat=False,
    fn="./analysis/Processed_data/Processed_data_withmap.csv",
    n_stim=6
        ):
    """Convert raw events data into meaningful trial data."""
    # Import assignment data for merge with trial data
    assignment = pd.read_csv(fp_assignment, sep=";", header=None)
    assignment.columns = [
        "pid", "age", "gender", "condition_id", "align_condition",
        "rotate_condition", "t_start", "browser", "platform", "screenHeight",
        "screenWidth", "status_code", "app_version", "x_axis",
        "creature_presence", "colour_0", "orientation_0"
        ]

    # Import events data
    df = pd.read_csv(fp_events, sep=";", header=None)
    df.columns = [
        "pid", "event_type", "t", "trial_no", "trial_id", "event_detail"
        ]
    df = df.drop_duplicates()
    df["block"] = np.floor(df["trial_no"]/n_stim)

    # Check that participants get the right number of unsupervised trials
    count_unsup = df.copy()

    count_unsup = count_unsup.loc[count_unsup["event_type"] == "unsupervised"]
    count_unsup = count_unsup[
                                ["pid", "event_type"]
                             ].groupby("pid").agg("count").reset_index()
    count_unsup.columns = ["pid", "nUnsupervised"]

    # Trial by participant df
    # Filter for each participant's timestamps above first trial load
    get_ti = df.groupby("pid").apply(get_all_trials)
    get_ti = get_ti.drop("pid", axis=1)
    get_ti.reset_index(inplace=True)

    # Filter out feedback clicks
    idx = [
            ("feedback-click" in str(x)) == False
            for x in get_ti["event_detail"]
          ]
    get_ti = get_ti.loc[idx]

    # Record trial number and stimulus ID
    columns = ["pid", "trial_no", "trial_id", "block"]
    trial_info = get_ti[columns].drop_duplicates()

    # Make column for trial type
    columns = ["pid", "trial_no", "trial_id", "event_detail"]
    t_types = get_ti.loc[
        (get_ti['event_type'] == "load") & (
            (get_ti['event_detail'] == "trial-generalisation") |
            (get_ti['event_detail'] == "trial-mainTrial")
        )
    ][columns].drop_duplicates()
    trial_info = trial_info.merge(
            t_types,
            on=["pid", "trial_no", "trial_id"]
        ).rename(
            columns={"event_detail": "trial_type"}
        )

    # Calculate time from page load to sumbission
    trial_times = get_ti.loc[
         ((get_ti['event_type'] == "load") &
          ((get_ti['event_detail'] == "trial-mainTrial") |
           (get_ti['event_detail'] == "trial-generalisation"))) |
         (get_ti['event_type'] == "submit")
        ].sort_values(by="t")
    trial_times_load = trial_times.loc[trial_times["event_type"] == "load"].drop_duplicates(
        subset=["pid", "event_type", "trial_no"], keep="first"
        )
    trial_times_sub = trial_times.loc[trial_times["event_type"] == "submit"].drop_duplicates(
        subset=["pid", "event_type", "trial_no"], keep="last"
        )
    trial_times = pd.concat([trial_times_load, trial_times_sub])
    trial_times = trial_times.pivot(
        values="t", index=["pid", "trial_no"], columns="event_type"
        )
    trial_times["duration"] = trial_times["submit"] - trial_times["load"]
    trial_times.reset_index(inplace=True)
    trial_times = trial_times.rename(columns={"load": "first_load"})
    trial_info = trial_info.merge(trial_times, on=["pid", "trial_no"])

    # Add nUnsupervised
    trial_info = trial_info.merge(count_unsup, on="pid", how="left").fillna(0)

    # Record response accuracy
    trial_accuracy = get_ti.loc[
            get_ti['event_type'] == "submit"
        ].sort_values(by="t")[
            ["pid", "trial_no", "event_detail"]
        ]
    trial_accuracy = trial_accuracy.dropna()
    trial_accuracy = trial_accuracy.rename(
        columns={"event_detail": "correct"}
        )
    trial_accuracy =trial_accuracy.drop_duplicates(
        subset=["pid", "trial_no"], keep="last"
        )
    trial_accuracy["correct"].loc[trial_accuracy['correct'] == 'correct'] = 1
    trial_accuracy["correct"].loc[trial_accuracy['correct'] == 'incorrect'] = 0
    trial_info = trial_info.merge(trial_accuracy, on=["pid", "trial_no"])

    # Record number of house clicks before submission
    trial_clicks = get_ti.loc[
            (get_ti['event_type'] == "click")
        ].sort_values(by="t")
    trial_clicks = trial_clicks[
            ['pid', 'trial_no', 'event_type']
        ].groupby(['pid', 'trial_no']).agg("count")
    trial_clicks.reset_index(inplace=True)
    trial_clicks = trial_clicks.rename(columns={"event_type": "box_clicks"})
    trial_info = trial_info.merge(trial_clicks, on=["pid", "trial_no"])

    # Record final house click before submission
    trial_clicks = get_ti.loc[
            (get_ti['event_type'] == "click") &
            (get_ti['event_detail'] != "zero-shot")
            ].sort_values(by="t")
    # Get rid of clicks where item clicked is not house
    trial_clicks["not_num"] = [
        str(x).isnumeric() is False for x in trial_clicks["event_detail"]
        ]
    trial_clicks = trial_clicks.loc[trial_clicks["not_num"] == True]
    trial_clicks["event_rank"] = trial_clicks[
        ['pid', 'trial_no', 'event_type', 't']
        ].groupby(['pid', 'trial_no', 'event_type']).t.transform('rank')
    trial_clicks["max_rank"] = trial_clicks[
            ['pid', 'trial_no', 'event_type', 'event_rank']
        ].groupby(
            ['pid', 'trial_no', 'event_type']
        ).event_rank.transform('max')
    ismax = trial_clicks["max_rank"] == trial_clicks["event_rank"]
    last_clicks = trial_clicks.loc[ismax][['pid', 'trial_no', 'event_detail']]
    last_clicks = last_clicks.rename(
        columns={'event_detail': 'submitted_house'}
        )
    trial_info = trial_info.merge(last_clicks, on=["pid", 'trial_no'])

    # Flag number of page blurs and total blur duration
    if "blur" in list(set(get_ti["event_type"])):
        trial_blurs = get_ti.loc[
            (get_ti['event_type'].isin(["focus", "blur"]))
            ].sort_values(by="t")
        trial_blurs["event_rank"] = trial_blurs[
            ['pid', 'trial_no', 'event_type', 't']
            ].groupby(['pid', 'event_type']).t.transform('rank')
        trial_blurs = trial_blurs.pivot(
            values="t",
            index=["pid", "trial_no", "event_rank"], columns="event_type"
            )
        trial_blurs.reset_index(inplace=True)
        trial_blurs["blur_duration"] = (
            trial_blurs["focus"] - trial_blurs["blur"]
        )
        trial_blurs = trial_blurs.drop(["focus", "event_rank"], axis=1)
        trial_blurs = trial_blurs.groupby(
            ["pid", "trial_no"]
            ).agg({"blur": "count", "blur_duration": "sum"})
        trial_blurs = trial_blurs.rename(
            columns={
                "blur": "blurs",
                "blur_duration": "total_blur_duration"
                }
                )
        trial_blurs.reset_index(inplace=True)
        trial_info = trial_info.merge(
            trial_blurs,
            on=["pid", "trial_no"],
            how="left").fillna(0, downcast='infer')

    # Flag number of page reloads/backs
    trial_loads = get_ti.loc[
                ((get_ti['event_type'] == "load") &
                 (get_ti['event_detail'] == "trial-mainTrial")) |
                ((get_ti['event_type'] == "load") &
                 (get_ti['event_detail'] == "trial-generalisation"))
            ].sort_values(by="t")
    trial_loads = trial_loads[
        ['pid', 'trial_no', 'event_type']
        ].groupby(['pid', 'trial_no']).agg("count")
    trial_loads.reset_index(inplace=True)
    trial_loads = trial_loads.rename(columns={"event_type": "page_loads"})
    trial_info = trial_info.merge(trial_loads, on=["pid", "trial_no"])

    # Flag if trial feedback was not played to completion
    trial_completes = get_ti.loc[
        (get_ti['event_type'] == "complete")
        ].sort_values(by="t")
    trial_completes = trial_completes[
        ['pid', 'trial_no', 'event_type']
        ].groupby(['pid', 'trial_no']).agg("count")
    trial_completes.reset_index(inplace=True)
    trial_completes = trial_completes.rename(
        columns={"event_type": "complete"}
        )
    trial_info = trial_info.merge(
                    trial_completes,
                    on=["pid", "trial_no"],
                    how="left"
                ).fillna(0, downcast='infer')
    trial_info["complete"].loc[
        trial_info["trial_type"] == "trial-generalisation"
        ] = 1

    # Merge participant info with trial info
    full_data = assignment.merge(trial_info, on="pid", how="left")

    # Import map data
    map_data = pd.read_csv(fp_map, header=None, sep=";")
    map_data.columns = ["pid", "houseID", "house_x", "house_y", "creature_x",
                        "creature_y", "trialID", "trial_type", "position_type"]
    map_data = map_data.drop_duplicates()

    # Normalise between 0 and 1
    for col in ["house_x", "house_y", "creature_x", "creature_y"]:
        map_data[col+"_normed"] = [float(x)-1 if x != "N" else "N" for x in map_data[col]]
        map_data[col+"_normed"] = [float(x)/5 if x != "N" else "N" for x in map_data[col]]

    # Get correct coordinates and creature space coordinates by trial ID
    correct = map_data[["pid", "house_x", "house_y", "house_x_normed", "house_y_normed", "creature_x",
                        "creature_y", "houseID", "trialID", "position_type"]]
    correct = correct.loc[correct["houseID"] != "zeroshot-incorrect"]
    correct = correct.rename(columns={"house_x": "correct_x",
                                      "house_y": "correct_y",
                                      "houseID": "correct_house",
                                      "trialID": "trial_id",
                                      "house_x_normed": "correct_x_normed",
                                      "house_y_normed": "correct_y_normed"})
    correct["trial_id"] = pd.to_numeric(correct["trial_id"])
    full_data = full_data.merge(correct, on=["pid", "trial_id"])

    # Get sumbitted coordinates by submitted_house
    submitted = map_data[["pid", "house_x", "house_y", "house_x_normed", "house_y_normed", "houseID"]]

    submitted = submitted.rename(columns={"house_x": "submitted_x",
                                          "house_y": "submitted_y",
                                          "house_x_normed": "submitted_x_normed",
                                          "house_y_normed": "submitted_y_normed",
                                          "houseID": "submitted_house"})

    full_data = full_data.merge(submitted, on=["pid", "submitted_house"])
    full_data = distance_from_correct(full_data)

    resident_creature = map_data[
                        ["pid", "creature_x", "creature_y", "houseID"]
                        ]
    resident_creature = resident_creature.rename(
                                columns={"creature_x": "resident_x",
                                         "creature_y": "resident_y",
                                         "houseID": "submitted_house"}
                                        )
    full_data = full_data.merge(
        resident_creature, on=["pid", "submitted_house"]
        )

    columns = ["pid",
               "house_x",
               "house_y",
               "creature_x",
               "creature_y",
               "position_type"]

    map_pos = map_data.loc[map_data["trial_type"] == "main"]
    alignment_scores = map_pos[columns].drop_duplicates().groupby(
                                    "pid"
                                    ).apply(
                                participant_wise_alignment
                                        )
    alignment_scores = alignment_scores.to_frame(
                            "alignment_correlation_trialstim"
                            ).reset_index()
    full_data = full_data.merge(alignment_scores, on="pid")
    full_data = full_data.drop_duplicates()

    alignment_scores = map_pos[columns].drop_duplicates().groupby(
                                    "pid"
                                    ).apply(
                                participant_wise_alignment, stimuli="all"
                                        )
    alignment_scores = alignment_scores.to_frame(
                            "alignment_correlation_allstim"
                            ).reset_index()
    full_data = full_data.merge(alignment_scores, on="pid")
    full_data = full_data.drop_duplicates()
    full_data.loc[
        full_data["creature_presence"] == False,
        "alignment_correlation_allstim"
        ] = "None"
    full_data["correct"] = pd.to_numeric(full_data["correct"])

    full_data = get_response_entropy(full_data, entropy="by_block", exclude_final=False)

    full_data.to_csv(
        fn
        )
    print("Processed data saved to file")

    if return_dat:
        return full_data

