#  Copyright (C) 2015 Larry Friedman, Brandeis University
#  Copyright (C) 2021-2022 Tzu-Yu Lee, National Taiwan University
#
#  This file (mapping.py) is part of blink.
#
#  blink is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  blink is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with blink.  If not, see <https://www.gnu.org/licenses/>.

"""
This module handles mapping operations, which is the image registration process
done via transforming AOI coordinates.
"""
import itertools
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyswarms
import scipy.io as sio

import blink.image_processing as imp

MapDirection = namedtuple("MapDirection", ["from_channel", "to_channel"])
DIR_DICT = {"r": "red", "g": "green", "b": "blue"}
AVAILABLE_CHANNELS = ("red", "green", "blue", "ir")


class Mapper:
    def __init__(self):
        self.map_matrix = None

    @classmethod
    def from_imscroll(cls, paths):
        mapper = cls()
        mapper.map_matrix = dict()
        if isinstance(paths, Path):
            paths = [paths]
        for path in paths:
            file_name = str(path.stem)
            match = re.search("_[rgb]{2}_", file_name)
            if match:
                direction_str = file_name[match.start() + 1 : match.end() - 1]
                direction = MapDirection(*(DIR_DICT[i] for i in direction_str))
            else:
                raise ValueError(
                    "Mapping file name does not provide direction information."
                )
            mapping_points = sio.loadmat(path)["mappingpoints"]
            mapper.map_matrix[direction] = make_map_matrix(
                mapping_points[:, [3, 2]] - 1, mapping_points[:, [9, 8]] - 1
            )
        return mapper

    def map(self, aois, to_channel: str):
        if aois.channel not in AVAILABLE_CHANNELS:
            raise ValueError("From-channel is not one of the available channels")
        if to_channel not in AVAILABLE_CHANNELS:
            raise ValueError("To-channel is not one of the available channels")
        if aois.channel == to_channel:
            aois_copy = imp.Aois(
                aois._coords,
                frame=aois.frame,
                frame_avg=aois.frame_avg,
                width=aois.width,
                channel=to_channel,
            )
            return aois_copy

        direction = MapDirection(from_channel=aois.channel, to_channel=to_channel)
        inv_direction = MapDirection(from_channel=to_channel, to_channel=aois.channel)
        if direction in self.map_matrix:
            map_matrix = self.map_matrix[direction]
        elif inv_direction in self.map_matrix:
            map_matrix = self._inverse_map_matrix(self.map_matrix[inv_direction])
        else:
            raise ValueError(
                f"Mapping matrix from channel {aois.channel}"
                f" to channel {to_channel} is not loaded"
            )

        new_coords = affine_transform(
            map_matrix[:, :2], map_matrix[:, 2], aois._coords.T
        )
        
        mapped_aois = imp.Aois(
            new_coords.T,
            frame=aois.frame,
            frame_avg=aois.frame_avg,
            width=aois.width,
            channel=to_channel,
        )
        return mapped_aois

    @staticmethod
    def _inverse_map_matrix(map_matrix):
        inv_A = np.linalg.inv(map_matrix[:, :2])
        inv_b = np.matmul(-inv_A, map_matrix[:, 2, np.newaxis])
        inv_map_matrix = np.concatenate((inv_A, inv_b), axis=1)
        return inv_map_matrix

    @classmethod
    def from_npz(cls, path: Path):
        mapper = cls()
        mapper.map_matrix = dict()
        with np.load(path) as npz_file:
            for file_name in npz_file.files:
                direction = MapDirection(*file_name.split("-"))
                mapper.map_matrix[direction] = make_map_matrix(
                    npz_file[file_name][:, :2], npz_file[file_name][:, 2:4]
                )
        return mapper


def make_map_matrix(x1y1, x2y2):
    map_matrix = np.vstack(
        (
            double_linear_regression(x2y2[:, 0], x1y1),
            double_linear_regression(x2y2[:, 1], x1y1),
        )
    )
    return map_matrix


def double_linear_regression(y, x1x2):
    X = np.hstack((x1x2, np.ones((x1x2.shape[0], 1))))
    beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    return beta.squeeze()


class MapperBare:
    def __init__(self):
        self.map_matrix = None

    @classmethod
    def from_imscroll(cls, path):
        mapper = cls()
        mapper.map_matrix = dict()
        mapping_points = sio.loadmat(path)["mappingpoints"]
        mapper.map_matrix = make_map_matrix(
            mapping_points[:, [3, 2]] - 1, mapping_points[:, [9, 8]] - 1
        )
        return mapper

    def map(self, aois):
        map_matrix = self.map_matrix
        new_coords = affine_transform(
            map_matrix[:, :2], map_matrix[:, 2], aois._coords.T
        )
        mapped_aois = imp.Aois(
            new_coords.T,
            frame=aois.frame,
            frame_avg=aois.frame_avg,
            width=aois.width,
        )
        return mapped_aois

    def inverse_map(self, aois):
        map_matrix = self._inverse_map_matrix(self.map_matrix)
        new_coords = affine_transform(
            map_matrix[:, :2], map_matrix[:, 2], aois._coords.T
        )
        mapped_aois = imp.Aois(
            new_coords.T,
            frame=aois.frame,
            frame_avg=aois.frame_avg,
            width=aois.width,
        )
        return mapped_aois

    @staticmethod
    def _inverse_map_matrix(map_matrix):
        inv_A = np.linalg.inv(map_matrix[:, :2])
        inv_b = np.matmul(-inv_A, map_matrix[:, 2, np.newaxis])
        inv_map_matrix = np.concatenate((inv_A, inv_b), axis=1)
        return inv_map_matrix

    @classmethod
    def from_npz(cls, path: Path):
        mapper = cls()
        with np.load(path) as npz_file:
            if len(npz_file.files) != 1:
                raise ValueError(
                    "The number of channel combinations in the mapping file "
                    "is larger than 1."
                )
            file_name = npz_file.files[0]
            mapper.map_matrix = make_map_matrix(
                npz_file[file_name][:, :2], npz_file[file_name][:, 2:4]
            )
        return mapper


def affine_transform(A, b, x):
    if b.shape == (2,):
        b = b[:, np.newaxis]
    y = np.matmul(A, x) + b
    return y


def _dist_arr(coords_a, coords_b):
    delta_x = coords_a[:, 0, np.newaxis] - coords_b[np.newaxis, :, 0]
    delta_y = coords_a[:, 1, np.newaxis] - coords_b[np.newaxis, :, 1]
    dist = np.sqrt(delta_x**2 + delta_y**2)
    return dist


def _partial_hausdorff_distance(k, A, B):
    dist = _dist_arr(A, B)
    dphd_a_b = np.sort(np.min(dist, axis=1))[k]
    dphd_b_a = np.sort(np.min(dist, axis=0))[k]
    return max(dphd_a_b, dphd_b_a)


def find_paired_spots_with_unknown_translation(
    spots_a: np.ndarray,
    spots_b: np.ndarray,
    bounds=(np.array([-512, -512]), np.array([512, 512])),
    options=None,
    d=1.5,
):
    """
    Find matched pairs of spots connected by an unknown translation tranform.

    In two groups of spots (A with size M, B with size N), assuming there exists K spots
    in each group that are paired by an unknown translation transform with some noise.
    This function will try to find the indices that selects this K pairs from A and B.
    The translation is found by optimizing a distance metric (partial Hausdorff distance
    (Huttenlocher, 1993)) by particle swarm optimization. Then unique point pairs that
    matches interms of distance falling withing a threshold d were found.

    The quantile for the partial Hausdorff distance is assumed to be min(M/2, N/2).
    This method is inspired by Yin (2006).
    doi: 10.1016/j.jvcir.2005.02.002

    Args:
        spots_a: A (M, 2) array holding the x, y coordinates of M spots of group A.
        spots_b: A (N, 2) array holding the x, y coordinates of N spots of group B.
        bounds: The bounds of the translation parameter to be optimized. It should be an
                tuple with len == 2. Each element is a ndarray as [lower, upper]. Passed
                into pyswarms.
        options: The particle swarm parameters. See pyswarms documentation.
        d: The distance threshold for recognizing pairs.

    Returns:
        matched_pairs: A (K, 2) array holding the indices to select the found K pairs.
                       A row with value [i, j] means that the ith spot of group A and
                       the jth spot of group B is a pair.
    """
    k = round(min(spots_a.shape[0], spots_b.shape[0]) / 2)

    def loss(param):
        # Need a loop because pyswarms seems to evaluate the function for all
        # particles at once. Therefore, param is a 2-D array, with the first
        # axis being particles and second axis being the dimension of parameter
        # space.
        phds = []
        for dx, dy in param:
            transformed_spots_a = spots_a + np.array([[dx, dy]])
            phd = _partial_hausdorff_distance(k, transformed_spots_a, spots_b)
            phds.append(phd)
        return np.array(phds)

    if options is None:
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = pyswarms.single.GlobalBestPSO(
        n_particles=10, dimensions=2, options=options, bounds=bounds
    )
    _, translation = optimizer.optimize(loss, iters=1000, verbose=False)
    mapped_spot_a = spots_a + translation
    matched_pairs = _find_matched_pair(mapped_spot_a, spots_b, d)
    return matched_pairs


def _find_matched_pair(mapped_spot_a, spots_b, d):
    dist = _dist_arr(mapped_spot_a, spots_b)
    in_range = dist <= d
    matched_pairs = []
    # Find unique pairings that satisfies the distance threshold
    # So check for rows that has only one column distance lower than threshold
    for i, row in enumerate(in_range):
        if np.count_nonzero(row) == 1:
            matched_pairs.append((i, np.nonzero(row)[0].item()))
    return np.array(matched_pairs)


def make_mapping_from_directory_of_spots(
    source_dir: Path, save_path: Path, d: float = 3, d2: float = 0.6
):
    """
    Makes a mapping file from a set of AOIs files by finding matching pairs.

    The source_dir contains only the AOIs files. The AOIs files of the same field of
    view should have the same base name, and a channel string ('_r', '_g', '_b') should
    be appended at the end of the base name to specify channels. For example, my
    source_dir has three files ['0126_0_r.npz', '0126_0_g.npz', '0126_0_b.npz'].

    Args:
        source_dir: The path to the directory that contains the AOIs files.
        save_path: The path to save the mapping file.
        d: The distance threshold to find colocalized spots.
        d2: The distance threshold to refine the final mapping. Any pair with mapping
            error larger than this value will be iteratively removed.
    """
    aois_dict = {}
    for path in source_dir.iterdir():
        stem = path.stem[:-2]
        aois = imp.Aois.from_npz(path)
        channel = DIR_DICT[path.stem[-1]]
        if stem in aois_dict:
            aois_dict[stem][channel] = aois
        else:
            aois_dict[stem] = {channel: aois}

    map_matrix = {}
    spot_pairs = {}
    channels = None
    for spots in aois_dict.values():
        if channels is None:
            channels = sorted(spots.keys())
        for combination in itertools.combinations(channels, 2):
            channel1, channel2 = combination
            if combination not in spot_pairs:
                idx = find_paired_spots_with_unknown_translation(
                    spots[channel1].coords, spots[channel2].coords, d=d
                )
                spot_pairs[combination] = [
                    spots[channel1].coords[idx[:, 0]],
                    spots[channel2].coords[idx[:, 1]],
                ]
            else:
                mapped_ch1_coords = affine_transform(
                    map_matrix[combination][:, :2],
                    map_matrix[combination][:, 2],
                    spots[channel1].coords.T,
                ).T
                idx = _find_matched_pair(mapped_ch1_coords, spots[channel2].coords, d)
                spot_pairs[combination][0] = np.concatenate(
                    (spot_pairs[combination][0], spots[channel1].coords[idx[:, 0]]),
                    axis=0,
                )
                spot_pairs[combination][1] = np.concatenate(
                    (spot_pairs[combination][1], spots[channel2].coords[idx[:, 1]]),
                    axis=0,
                )
            map_matrix[combination] = make_map_matrix(*spot_pairs[combination])
    items_to_save = {}
    for combination in map_matrix.keys():
        spot_pairs[combination], map_matrix[combination] = _remove_inconsistent_pairs(
            spot_pairs[combination], map_matrix[combination], d2
        )
        items_to_save["{}-{}".format(*combination)] = np.concatenate(
            spot_pairs[combination], axis=1
        )
        ch1, ch2 = combination
        print(
            f"Collected {spot_pairs[combination][0].shape[0]} pairs "
            f"between channel {ch1} and {ch2}"
        )
        print(map_matrix[combination])
    np.savez(save_path, **items_to_save)


def _remove_inconsistent_pairs(spot_pairs, map_matrix, d):
    """
    Iteratively remove the pairs that the distance between mapped coordinates 1 and
    original coordinates 2 is above d.
    """
    diff = (
        affine_transform(map_matrix[:, :2], map_matrix[:, 2], spot_pairs[0].T).T
        - spot_pairs[1]
    )
    dist = np.sqrt(np.sum(diff**2, axis=1))
    while np.any(dist > d):
        idx = dist <= d
        spot_pairs[0] = spot_pairs[0][idx, :]
        spot_pairs[1] = spot_pairs[1][idx, :]
        map_matrix = make_map_matrix(*spot_pairs)
        diff = (
            affine_transform(map_matrix[:, :2], map_matrix[:, 2], spot_pairs[0].T).T
            - spot_pairs[1]
        )
        dist = np.sqrt(np.sum(diff**2, axis=1))
    return spot_pairs, map_matrix
