from pathlib import Path

import numpy as np
import pytest

import blink.drift_correction as dcorr
import blink.image_processing as imp

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_drift_corrector_collective_translation():
    n_frames = 100
    n_molecules = 100
    rng = np.random.default_rng(0)
    coords = np.zeros((n_frames, n_molecules, 2))
    # Coordinates start from 10 to avoid negative values
    coords[0, :, :] = rng.uniform(10, 100, size=(n_molecules, 2))
    for frame in range(1, n_frames):
        coords[frame, :, :] = coords[frame - 1, :, :] + rng.normal(
            loc=0.5 * frame, scale=1, size=2
        )
    drifter = dcorr.DriftCorrector.from_coords(
        coords, (0, n_frames), smooth=lambda x: x
    )
    for start in range(n_frames):
        # Make Aois with non-default attributes to check whether they are copied
        # correctly
        aois = imp.Aois(
            coords[start],
            start,
            frame_avg=2,
            width=6,
            channel=imp.Channel("green", "green"),
        )
        # TODO: The current behavior only allows to shift forwards
        # could implement both ways later
        for frame in range(start, n_frames):
            drifted = drifter.shift_aois(aois, frame)
            assert isinstance(drifted, imp.Aois)
            assert drifted.frame == frame
            assert drifted.frame_avg == aois.frame_avg
            assert drifted.width == aois.width
            assert drifted.channel == aois.channel
            np.testing.assert_allclose(drifted.coords, coords[frame, :, :])


def test_drift_corrector_collective_translation_partial_range():
    n_frames = 100
    frame_range = (20, 80)
    n_molecules = 100
    rng = np.random.default_rng(1)
    coords = np.zeros((frame_range[1] - frame_range[0], n_molecules, 2))
    # Coordinates start from 10 to avoid negative values
    coords[0, :, :] = rng.uniform(10, 100, size=(n_molecules, 2))
    for frame in range(1, frame_range[1] - frame_range[0]):
        coords[frame, :, :] = coords[frame - 1, :, :] + rng.normal(
            loc=0.5 * frame, scale=1, size=2
        )
    drifter = dcorr.DriftCorrector.from_coords(coords, frame_range, smooth=lambda x: x)
    for start in range(n_frames):
        # No coordinate shift for frames outside frame_range
        i = max(frame_range[0], start)
        i = min(i, frame_range[1] - 1)
        aois = imp.Aois(
            coords[i - frame_range[0]],
            start,
            frame_avg=2,
            width=6,
            channel=imp.Channel("green", "green"),
        )
        # TODO: The current behavior only allows to shift forwards
        # could implement both ways later
        for frame in range(start, n_frames):
            drifted = drifter.shift_aois(aois, frame)
            assert isinstance(drifted, imp.Aois)
            assert drifted.frame == frame
            assert drifted.frame_avg == aois.frame_avg
            assert drifted.width == aois.width
            assert drifted.channel == aois.channel
            # Shifting to outside of the frame_range should result in the coordinates
            # of the first frame or last frame in the range
            j = max(frame_range[0], frame)
            j = min(j, frame_range[1] - 1)
            np.testing.assert_allclose(drifted.coords, coords[j - frame_range[0], :, :])


def test_drift_corrector_from_coords_input_range():
    """Check for invalid input ranges"""
    lengths = [12, 50, 58, 100]
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    for length in lengths:
        coords = np.zeros((length, 10, 2))
        for a, b in directions:
            with pytest.raises(ValueError) as r:
                dcorr.DriftCorrector.from_coords(
                    coords, (a, length + b), smooth=lambda x: x
                )
            assert "Mismatched length" in str(r.value)


def test_drift_corrector_from_coords_invalid_coords():
    """Check for invalid input coordinates"""
    n_frames = 10
    # The dimension of coords should be 3
    cases = [np.zeros(n_frames), np.zeros((n_frames, 2)), np.zeros((n_frames, 1, 2, 3))]
    for coords in cases:
        with pytest.raises(ValueError) as r:
            dcorr.DriftCorrector.from_coords(coords, (0, n_frames), smooth=lambda x: x)
        message = str(r.value)
        assert "Invalid number of dimensions" in message
        assert "should be 3" in message

    cases = [
        np.zeros((n_frames, 1, 1)),
        np.zeros((n_frames, 1, 3)),
        np.zeros((n_frames, 1, 4)),
    ]
    for coords in cases:
        with pytest.raises(ValueError) as r:
            dcorr.DriftCorrector.from_coords(coords, (0, n_frames), smooth=lambda x: x)
        message = str(r.value)
        assert "is not 2" in message


def test_drift_corrector_load_save_npy():
    frame_range = (20, 80)
    n_molecules = 100
    rng = np.random.default_rng(0)
    coords = np.zeros((frame_range[1] - frame_range[0], n_molecules, 2))
    # Coordinates start from 10 to avoid negative values
    coords[0, :, :] = rng.uniform(10, 100, size=(n_molecules, 2))
    for frame in range(1, frame_range[1] - frame_range[0]):
        coords[frame, :, :] = coords[frame - 1, :, :] + rng.normal(
            loc=0.5 * frame, scale=1, size=2
        )
    drifter = dcorr.DriftCorrector.from_coords(coords, frame_range, smooth=lambda x: x)
    save_path = TEST_DATA_DIR / "save_path"
    drifter.to_npy(save_path / "driftlist.npy")
    loaded_drifter = dcorr.DriftCorrector.from_npy(save_path / "driftlist.npy")
    np.testing.assert_equal(loaded_drifter._driftlist, drifter._driftlist)


def test_shift_aois_by_time():
    driftlist = np.tile(np.arange(20)[:, np.newaxis], (1, 4))
    driftlist[1:, 1:3] = np.diff(driftlist[:, 1:3], axis=0)
    drifter = dcorr.DriftCorrector._from_diff_driftlist(driftlist)
    aois = imp.Aois(np.zeros((1, 2)), 0)
    for t in np.arange(0, 19.05, 0.05):
        new_aois = drifter.shift_aois_by_time(aois, t)
        np.testing.assert_equal(new_aois.coords, aois.coords + t)

    # Check for exception when the given time is outside of the correction
    # range
    # for t in [-1, -0.1, 19.05, 20]:
    #     with pytest.raises(ValueError) as exception_info:
    #         new_aois = drifter.shift_aois_by_time(aois, t)
    #     assert str(exception_info.value) == f'time {t} is not in drift correction range [0, 19]'
