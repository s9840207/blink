import io
from pathlib import Path

import numpy as np
import pytest

from blink import image_processing as imp
from blink import mapping

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_mapper_from_imscroll():
    path = TEST_DATA_DIR / "mapping/20200206_br_02.dat"
    mapper = mapping.Mapper.from_imscroll(path)
    assert isinstance(mapper, mapping.Mapper)
    assert ("blue", "red") in mapper.map_matrix

    aois = imp.Aois.from_imscroll_aoiinfo2(TEST_DATA_DIR / "mapping/L2_aoi.dat")
    aois.channel = "blue"
    for channel in ("red", "blue"):
        new_aois = mapper.map(aois, to_channel=channel)
        assert isinstance(new_aois, imp.Aois)
        assert new_aois.frame == aois.frame
        assert new_aois.frame_avg == aois.frame_avg
        assert new_aois.width == aois.width
        assert new_aois.channel == channel
        if channel == "red":
            correct_aois = imp.Aois.from_imscroll_aoiinfo2(
                TEST_DATA_DIR / "mapping/L2_map.dat"
            )
            np.testing.assert_allclose(
                new_aois._coords, correct_aois._coords, rtol=1e-6
            )

    with pytest.raises(ValueError) as exception_info:
        new_aois = mapper.map(aois, to_channel="green")
        assert exception_info.value == (
            "Mapping matrix from channel blue" " to channel green is not loaded"
        )

    for to_channel in ("black", 123):
        with pytest.raises(ValueError) as exception_info:
            new_aois = mapper.map(aois, to_channel=to_channel)
            assert (
                exception_info.value
                == "To-channel is not one of the available channels"
            )

    for from_channel in ("black", 123):
        aois = imp.Aois.from_imscroll_aoiinfo2(TEST_DATA_DIR / "mapping/L2_aoi.dat")
        aois.channel = from_channel
        with pytest.raises(ValueError) as exception_info:
            new_aois = mapper.map(aois, to_channel=to_channel)
            assert (
                exception_info.value
                == "From-channel is not one of the available channels"
            )

    from_channel = "red"
    channel = "blue"
    aois = imp.Aois.from_imscroll_aoiinfo2(TEST_DATA_DIR / "mapping/L2_aoi.dat")
    aois.channel = from_channel

    new_aois = mapper.map(aois, to_channel=channel)
    assert isinstance(new_aois, imp.Aois)
    assert new_aois.frame == aois.frame
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.width == aois.width
    assert new_aois.channel == channel
    correct_aois = imp.Aois.from_imscroll_aoiinfo2(
        TEST_DATA_DIR / "mapping/L2_inv_map.dat"
    )
    np.testing.assert_allclose(new_aois._coords, correct_aois._coords, rtol=1e-6)


def test_find_paired_spots_with_unknown_translation():
    rng = np.random.default_rng(seed=0)
    aois_a = imp.Aois(rng.uniform(high=500, size=(100, 2)), 0)
    aois_a = aois_a.remove_close_aois(5)
    spots_a = aois_a.coords
    spots_b = np.zeros((100, 2))
    spots_b[:70, :] = spots_a[:70, :] + np.array([[200, 200]])
    # Need to make sure all spots are separated by at least 5 pixels
    for i in range(70, 100):
        while True:
            new_aoi = imp.Aois(rng.uniform(low=200, high=700, size=(1, 2)), 0)
            if not new_aoi.is_in_range_of(aois_a, 5):
                break
        spots_b[i, :] = new_aoi.coords
    pair_idx = mapping.find_paired_spots_with_unknown_translation(spots_a, spots_b)
    np.testing.assert_equal(pair_idx, np.stack((np.arange(70), np.arange(70)), axis=1))


def test_find_paired_spots_with_unknown_translation_reversed():
    rng = np.random.default_rng(seed=0)
    aois_a = imp.Aois(rng.uniform(high=500, size=(100, 2)), 0)
    aois_a = aois_a.remove_close_aois(5)
    spots_a = aois_a.coords
    spots_b = np.zeros((100, 2))
    spots_b[:70, :] = spots_a[:70, :] + np.array([[200, 200]])
    # Need to make sure all spots are separated by at least 5 pixels
    for i in range(70, 100):
        while True:
            new_aoi = imp.Aois(rng.uniform(low=200, high=700, size=(1, 2)), 0)
            if not new_aoi.is_in_range_of(aois_a, 5):
                break
        spots_b[i, :] = new_aoi.coords
    # Reversing the order of b
    spots_b = np.flip(spots_b, axis=0)
    pair_idx = mapping.find_paired_spots_with_unknown_translation(spots_a, spots_b)
    np.testing.assert_equal(
        pair_idx, np.stack((np.arange(70), 99 - np.arange(70)), axis=1)
    )


def test_find_paired_spots_with_unknown_translation_with_noise():
    rng = np.random.default_rng(seed=0)
    aois_a = imp.Aois(rng.uniform(high=500, size=(100, 2)), 0)
    aois_a = aois_a.remove_close_aois(5)
    spots_a = aois_a.coords
    spots_b = np.zeros((100, 2))
    spots_b[:70, :] = spots_a[:70, :] + np.array([[200, 200]])
    # Add noise to transformed coordinates
    spots_b += rng.normal(scale=0.4, size=spots_b.shape)
    # Need to make sure all spots are separated by at least 5 pixels
    for i in range(70, 100):
        while True:
            new_aoi = imp.Aois(rng.uniform(low=200, high=700, size=(1, 2)), 0)
            if not new_aoi.is_in_range_of(aois_a, 5):
                break
        spots_b[i, :] = new_aoi.coords
    pair_idx = mapping.find_paired_spots_with_unknown_translation(spots_a, spots_b)
    np.testing.assert_equal(pair_idx, np.stack((np.arange(70), np.arange(70)), axis=1))


def test_find_paired_spots_with_unknown_translation_with_noise_shuffled():
    rng = np.random.default_rng(seed=0)
    aois_a = imp.Aois(rng.uniform(high=500, size=(100, 2)), 0)
    aois_a = aois_a.remove_close_aois(5)
    spots_a = aois_a.coords
    spots_b = np.zeros((100, 2))
    spots_b[:70, :] = spots_a[:70, :] + np.array([[200, 200]])
    # Add noise to transformed coordinates
    spots_b += rng.normal(scale=0.4, size=spots_b.shape)
    # Need to make sure all spots are separated by at least 5 pixels
    for i in range(70, 100):
        while True:
            new_aoi = imp.Aois(rng.uniform(low=200, high=700, size=(1, 2)), 0)
            if not new_aoi.is_in_range_of(aois_a, 5):
                break
        spots_b[i, :] = new_aoi.coords
    # Shuffles spots_b
    shuffle = np.arange(100)
    rng.shuffle(shuffle)
    spots_b = spots_b[shuffle, :]
    pair_idx = mapping.find_paired_spots_with_unknown_translation(spots_a, spots_b)
    np.testing.assert_equal(
        pair_idx, np.stack((np.arange(70), np.argsort(shuffle)[:70]), axis=1)
    )


def test_mapper_from_npz():
    map_matrix = np.array([[1, 0, 100], [0, 1, 10]])
    rng = np.random.default_rng(seed=0)
    coords1 = rng.uniform(high=500, size=(100, 2))
    coords2 = mapping.affine_transform(map_matrix[:, :2], map_matrix[:, 2], coords1.T).T
    f = io.BytesIO()
    items = {"green-red": np.concatenate((coords1, coords2), axis=1)}
    np.savez(f, **items)
    f.flush()
    f.seek(0)
    mapper = mapping.Mapper.from_npz(f)
    np.testing.assert_allclose(
        mapper.map_matrix[("green", "red")], map_matrix, rtol=0, atol=1e-11
    )


def test_mapper_bare_from_npz():
    map_matrix = np.array([[1, 0, 100], [0, 1, 10]])
    rng = np.random.default_rng(seed=0)
    coords1 = rng.uniform(high=500, size=(100, 2))
    coords2 = mapping.affine_transform(map_matrix[:, :2], map_matrix[:, 2], coords1.T).T
    f = io.BytesIO()
    items = {"green-red": np.concatenate((coords1, coords2), axis=1)}
    np.savez(f, **items)
    f.flush()
    f.seek(0)
    mapper = mapping.MapperBare.from_npz(f)
    np.testing.assert_allclose(mapper.map_matrix, map_matrix, rtol=0, atol=1e-11)
