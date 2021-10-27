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
import pandas as pd


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

