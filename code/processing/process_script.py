#! /opt/anaconda3/envs/align/bin/python

from os.path import dirname, abspath, join
import sys

import process_data as process
from process_utils import plot_block_means_2conditions

d = dirname(abspath(__file__))
sys.path.append(
    d
    )
sys.path.append(
    dirname(d)
    )

# Processing function - extract trial info from raw events file
dat = process.process_raw_data(
    fp_assignment=join(
                        dirname(dirname(d)),
                        "data/experiment/raw/all_assignments.csv"
                       ),
    fp_events=join(
                   dirname(dirname(d)),
                   "data/experiment/raw/all_events.csv"
                   ),
    fp_map=join(
                dirname(dirname(d)),
                "data/experiment/raw/all_maps.csv"
                ),
    return_dat=True,
    fn=join(
            dirname(dirname(d)),
            "data/experiment/processed/Processed_data.csv"
            ),
    n_stim=6
    )

# Apply exclusions
dat = process.apply_filters(dat, entropy=True)
dat.to_csv(join(
                dirname(dirname(d)),
                "data/experiment/processed/Processed_data_filtered.csv"
                )
           )


# Plotting script for results plots - percent accuracy and distance error

plot_block_means_2conditions(
    dat,
    col_name="correct",
    unit="block"
    )

plot_block_means_2conditions(
    dat,
    col_name="dist_from_correct",
    unit="block"
    )
