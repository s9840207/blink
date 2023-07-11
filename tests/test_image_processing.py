#!/usr/bin/env python3
"""

Add a fake image as test data. The faked image is generated as 400x600
array, with 20 frames. Each frame is consecutive integer from the frame
number (starting from 1) to frame number + 400x600 -1, reshaped row by
row.

"""
import pathlib

import numpy as np
import pytest
import scipy.io as sio

from blink import image_processing as imp

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_data"


def test_read_glimpse_image():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im/"
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame in range(image_sequence.length):
        image = image_sequence.get_one_frame(i_frame)
        assert image.shape == (image_sequence.height, image_sequence.width)
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, image)

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(-1)
        assert exception_info.value == "Frame number must be positive integers or 0"

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame("")
        assert exception_info.value == "Frame number must be positive integers or 0"

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(image_sequence.length)
        assert (
            exception_info.value
            == "Frame number ({}) exceeds sequence length - 1 ({})".format(
                image_sequence.length, image_sequence.length - 1
            )
        )


def test_image_sequence_class():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im/"
    image_sequence = imp.ImageSequence(image_path)
    assert image_sequence.width == 300
    assert image_sequence.height == 200
    assert image_sequence.length == 20


def test_iter_over_image_sequece():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im/"
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame, i_frame_image in enumerate(image_sequence):
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, i_frame_image)


def test_slicing_image_sequece():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im/"
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height

    for i_frame, i_frame_image in zip([2], image_sequence[2]):
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, i_frame_image)

    for i_frame, i_frame_image in zip([19], image_sequence[-1:]):
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, i_frame_image)

    for i_frame, i_frame_image in zip(range(2, 10, 2), image_sequence[2:10:2]):
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, i_frame_image)

    frames = [2, 4, 5, 7]
    for i_frame, i_frame_image in zip(frames, image_sequence[frames]):
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, i_frame_image)


def test_band_pass():
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_image.mat")
    test_image = sio.loadmat(image_path)["testImage"]
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_bpass.mat")
    true_image = sio.loadmat(image_path)["filteredImage"]
    filtered_image = imp.band_pass(test_image, 1, 5)
    np.testing.assert_allclose(true_image, filtered_image, atol=1e-12)


def test_band_pass_real_image():
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_bpass_real_image.mat")
    test_image = sio.loadmat(image_path)["image"]
    true_image = sio.loadmat(image_path)["filteredImage"]
    filtered_image = imp.band_pass(test_image, 1, 5)
    np.testing.assert_allclose(true_image, filtered_image, atol=1e-12)


def test_find_peak():
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_pkfnd_71_5.mat")
    test_image = sio.loadmat(image_path)["filteredImage"]
    true_peaks = sio.loadmat(image_path)["spotCoords"]
    peaks = imp.find_peaks(test_image, threshold=71, peak_size=5)
    assert isinstance(peaks, np.ndarray)
    assert peaks.shape[1] == 2
    assert np.issubdtype(
        peaks.dtype, np.integer
    )  # Check that the returned type is some integer
    peaks += 1  # Convert to 1 based indexing
    np.testing.assert_equal(peaks, true_peaks)

    # Case2 with different param
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_pkfnd_1_5.mat")
    test_image = sio.loadmat(image_path)["filteredImage"]
    true_peaks = sio.loadmat(image_path)["spotCoords"]
    peaks = imp.find_peaks(test_image, threshold=1, peak_size=5)
    assert isinstance(peaks, np.ndarray)
    assert peaks.shape[1] == 2
    assert np.issubdtype(
        peaks.dtype, np.integer
    )  # Check that the returned type is some integer
    peaks += 1  # Convert to 1 based indexing
    np.testing.assert_equal(peaks, true_peaks)


def test_localize_centroid():
    image_path = pathlib.Path(TEST_DATA_DIR / "spot_picking/test_cntrd_71_5.mat")
    test_image = sio.loadmat(image_path)["filteredImage"]
    peaks = (
        sio.loadmat(image_path)["spotCoords"] - 1
    )  # Minus 1 to convert to 0 based index
    true_output = (
        sio.loadmat(image_path)["out"][:, 0:2] - 1
    )  # First two columns are x, y coords
    output = imp.localize_centroid(test_image, peaks, 5 + 2)
    assert isinstance(output, np.ndarray)
    assert output.shape[1] == 2
    np.testing.assert_allclose(
        output, true_output, atol=1e-13
    )  # Tolerate rounding error

    # If there is no peak found by find_peaks()
    output = imp.localize_centroid(test_image, None, 5 + 2)
    assert output is None


def test_pick_spots():
    test_image = np.zeros((200, 300))
    aois = imp.pick_spots(test_image, noise_dia=1, spot_dia=5, threshold=50)
    assert isinstance(aois, imp.Aois)


def test_aois_class():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    assert aois.width == 5
    assert aois.frame == 0
    assert aois.frame_avg == 1
    true_arr = np.arange(10)
    np.testing.assert_equal(aois.get_all_x(), true_arr)
    np.testing.assert_equal(aois.get_all_y(), true_arr)

    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0, frame_avg=10, width=6)
    assert aois.width == 6
    assert aois.frame_avg == 10

    # These two attributes are protected by property, should not be set outside
    # init
    with pytest.raises(AttributeError) as exception:
        aois.frame = 20
        assert "can't set attribute" in str(exception.value)
    with pytest.raises(AttributeError):
        aois.frame_avg = 1
        assert "can't set attribute" in str(exception.value)

    assert len(aois) == 10

    # Iterator returns tuples of (x, y)
    for i, item in enumerate(aois):
        assert isinstance(item, tuple)
        x, y = item
        assert x == i
        assert y == i

    a = np.arange(10)
    aois = imp.Aois(np.stack([a, 2 * a], axis=-1), frame=1)

    np.testing.assert_equal(aois.get_all_x(), a)
    np.testing.assert_equal(aois.get_all_y(), a * 2)

    for i, item in enumerate(aois):
        assert isinstance(item, tuple)
        x, y = item
        assert x == i
        assert y == 2 * i

    assert (1, 2) in aois
    assert np.array([1, 2]) in aois
    assert (1, 1) not in aois
    assert (1, 1, 1) not in aois

    old_coords = aois.coords
    new_aois = aois + (105.2, 50)
    np.testing.assert_equal(new_aois.coords[:-1, :], old_coords)
    np.testing.assert_equal(new_aois.coords[-1, :], np.array((105.2, 50)))


def test_aoi_coords_validation():
    coords_list = [np.array([[0, 1], [-1, 2]]), np.array([[1, -1], [3, 0]])]
    for coords in coords_list:
        with pytest.raises(ValueError) as exception_info:
            imp.Aois(coords, 0)
            assert (
                exception_info.value
                == "Coordinates of an AOI should be positive valued or 0."
            )

    # Check the ndim of the coords array
    with pytest.raises(ValueError) as exception_info:
        imp.Aois(np.zeros(2), 0)
        assert (
            exception_info.value
            == "The ndim of the coordinates array should be 2. Instead we have 1."
        )

    with pytest.raises(ValueError) as exception_info:
        imp.Aois(np.zeros((10, 2, 1)), 0)
        assert (
            exception_info.value
            == "The ndim of the coordinates array should be 2. Instead we have 3."
        )


def test_remove_close_aois():
    # x spacing
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4
    aois = imp.Aois(arr, 0, frame_avg=10, width=10)
    new_aois = aois.remove_close_aois(5)
    assert isinstance(new_aois, imp.Aois)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])
    assert new_aois.frame == 0
    assert new_aois.frame_avg == 10
    assert new_aois.width == 10

    # y spacing
    arr = np.ones((7, 2))
    arr[:, 1] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])

    # Boundary case
    arr = np.ones((4, 2))
    arr[:, 1] = np.array([1, 3, 4, 8]) * 5
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3]])

    # Not on axis
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4 * np.cos(np.pi / 3)
    arr[:, 1] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4 * np.sin(np.pi / 3)
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])

    # Aggregates
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 1, 2, 6, 11, 13, 15])
    arr[:, 1] = np.array([1, 2, 1, 6, 11, 13, 14])
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[3]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[3]])


def test_Aois_is_in_range_of():
    arr = np.array(
        [
            [0, 1],
            [10, 10],
            [100, 50],
            [100, 52],
            [100, 49],
            [200, 100],
            [300, 70],
            [400, 80],
            [786, 520],
            [150, 200],
        ]
    )
    ref = np.array([[5, 1], [100, 55], [10, 15], [302, 68], [403, 84], [790, 516]])
    aois = imp.Aois(arr, 0)
    ref_aois = imp.Aois(ref, 0)
    is_in_range = aois.is_in_range_of(ref_aois=ref_aois, radius=5)
    assert len(is_in_range) == len(aois)
    true_arr = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    np.testing.assert_equal(is_in_range, true_arr)

    # Remove aois near ref
    new_aois = aois.remove_aois_near_ref(ref_aois, radius=5)
    np.testing.assert_equal(new_aois.get_all_x(), arr[np.logical_not(true_arr), 0])

    # Remove aois far from ref
    new_aois = aois.remove_aois_far_from_ref(ref_aois, radius=5)
    np.testing.assert_equal(new_aois.get_all_x(), arr[true_arr, 0])


def test_fit_2d_gaussian():
    np.random.seed(1)
    xy_data = np.ogrid[:5, :5]
    xy_data[1] = xy_data[1].T
    param = [500, 2.5, 1.7, 1.5, 100]
    image = imp.symmetric_2d_gaussian(xy_data, *param) + 10 * np.random.standard_normal(
        25
    )
    fitted_param = imp.fit_2d_gaussian(xy_data, image)
    np.testing.assert_allclose(fitted_param, param, rtol=0.03)


def test_Aois_iter_objects():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    for i, aoi in enumerate(aois.iter_objects()):
        assert aoi.width == 5
        assert aoi.frame == 0
        assert aoi.frame_avg == 1
        assert len(aoi) == 1
        assert (aoi.coords == i).all()


def test_Aois_getitem():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    for i in range(10):
        aoi = aois[i]
        assert aoi.width == 5
        assert aoi.frame == 0
        assert aoi.frame_avg == 1
        assert len(aoi) == 1
        assert (aoi.coords == i).all()
        assert aoi.coords.shape == (1, 2)
    with pytest.raises(IndexError) as exception_info:
        aoi = aois[10]
    assert (
        str(exception_info.value) == "index 10 is out of bounds for Aois with length 10"
    )


def test_aoi_get_subimage_on_grid():
    height, width = 10, 20
    coords = np.zeros((width * height, 2))
    for x in range(width):
        for y in range(height):
            coords[x + y * width, :] = [x, y]
    aoi_width = 5
    aois = imp.Aois(coords, frame=0, width=aoi_width)

    # This function should only work with individual AOIs
    with pytest.raises(ValueError) as e:
        aois.get_subimage_slice()
        assert e.value == f"Wrong AOI length ({len(aois)}), must be 1."

    for aoi in aois.iter_objects():
        subimage_idx = aoi.get_subimage_slice()
        assert isinstance(subimage_idx, tuple)
        assert len(subimage_idx) == 2
        y_slice, x_slice = subimage_idx
        assert isinstance(x_slice, slice)
        assert isinstance(y_slice, slice)

        x, y = aoi.coords[0]
        x, y = int(x), int(y)
        # Need to account for the distance to the left and upper edge and a 0.5 rounding
        # tolerance
        within_left = x - (-0.5) >= aoi_width / 2 - 0.5
        within_top = y - (-0.5) >= aoi_width / 2 - 0.5
        if within_left:
            assert x_slice.stop - x_slice.start == aoi_width
            assert (x_slice.stop + x_slice.start) / 2 - 0.5 == x
        else:
            # Minus how much of AOI go over the edge
            assert x_slice.stop - x_slice.start == aoi_width - round(
                aoi_width / 2 - 0.5 - x
            )
        if within_top:
            assert y_slice.stop - y_slice.start == aoi_width
            assert (y_slice.stop + y_slice.start) / 2 - 0.5 == y
        else:
            assert y_slice.stop - y_slice.start == aoi_width - round(
                aoi_width / 2 - 0.5 - y
            )
        # TODO: this function is currently not aware of the width and height of the
        # image. Should also add checks.


def test_aoi_get_subimage():
    height, width = 10, 20
    rng = np.random.default_rng(0)
    n_aois = 1000
    coords = np.stack(
        (
            rng.uniform(size=n_aois, high=width),
            rng.uniform(size=n_aois, high=height),
        )
    ).T
    for aoi_width in [5, 6]:
        aois = imp.Aois(coords, frame=0, width=aoi_width)

        # This function should only work with individual AOIs
        with pytest.raises(ValueError) as e:
            aois.get_subimage_slice()
            assert e.value == f"Wrong AOI length ({len(aois)}), must be 1."

        for aoi in aois.iter_objects():
            subimage_idx = aoi.get_subimage_slice()
            assert isinstance(subimage_idx, tuple)
            assert len(subimage_idx) == 2
            y_slice, x_slice = subimage_idx
            assert isinstance(x_slice, slice)
            assert isinstance(y_slice, slice)

            x, y = aoi.coords[0]
            # Need to account for the distance to the left and upper edge and a 0.5 rounding
            # tolerance
            within_left = x - (-0.5) >= aoi_width / 2 - 0.5
            within_top = y - (-0.5) >= aoi_width / 2 - 0.5
            print(x, y, subimage_idx)
            if within_left:
                assert x_slice.stop - x_slice.start == aoi_width
                assert abs((x_slice.stop + x_slice.start) / 2 - x - 0.5) <= 0.5
            else:
                # Minus how much of AOI go over the edge
                assert x_slice.stop - x_slice.start == aoi_width - round(
                    aoi_width / 2 - 0.5 - x
                )
            if within_top:
                assert y_slice.stop - y_slice.start == aoi_width
                assert abs((y_slice.stop + y_slice.start) / 2 - y - 0.5) <= 0.5
            else:
                assert y_slice.stop - y_slice.start == aoi_width - round(
                    aoi_width / 2 - 0.5 - y
                )
            # TODO: this function is currently not aware of the width and height of the
            # image. Should also add checks.


def test_aoi_gaussian_refine():
    width, height = 512, 256
    rng = np.random.default_rng(0)
    n_aois = 200
    aoi_widths = [15, 14]  # Both odd and even AOI width should work
    sigma = 2
    max_width = max(aoi_widths)
    true_coords = np.stack(
        (
            rng.uniform(low=max_width, high=width - max_width, size=n_aois),
            rng.uniform(low=max_width, high=height - max_width, size=n_aois),
        )
    ).T
    # The generated Gaussian peaks should not overlap
    true_aois = imp.Aois(true_coords, 0, width=max_width).remove_close_aois(max_width)
    true_coords = true_aois.coords
    image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            for x, y in true_coords:
                denom = (x - j) ** 2 + (y - i) ** 2
                if denom > (4 * sigma) ** 2:
                    continue
                image[i, j] += np.exp(-denom / 2 / sigma**2)
    for aoi_width in aoi_widths:
        # If offset from the peak center, gaussian_refine should recover the peak
        # center coordinate
        coords = true_coords + rng.normal(scale=sigma / 2, size=true_coords.shape)
        aois = imp.Aois(coords, 0, width=aoi_width)
        aois = aois.gaussian_refine(image)
        np.testing.assert_allclose(aois.coords, true_coords, rtol=0, atol=1e-2)
        # If we start from the actual coordinates, the refine function should not do
        # too much modifications
        aois.coords = true_coords
        aois = aois.gaussian_refine(image)
        np.testing.assert_allclose(aois.coords, true_coords, rtol=0, atol=1e-5)


def test_get_averaged_image():
    # Check that the averaged image is withing rounding error
    image_path = pathlib.Path(__file__).parent / "test_data/20200228/hwligroup00774"
    image_sequence = imp.ImageSequence(image_path)
    averaged_image = image_sequence.get_averaged_image(start=0, size=10)
    true_path = pathlib.Path(__file__).parent / "test_data/20200228/averaged_im.mat"
    true_image = sio.loadmat(true_path)["currentFrameImage"].T
    np.testing.assert_allclose(averaged_image, true_image, atol=0.5)


def test_remove_aoi_nearest_to_ref():
    arr = np.array(
        [
            [0, 1],
            [10, 10],
            [100, 50],
            [100, 52],
            [100, 49],
            [200, 100],
            [300, 70],
            [400, 80],
            [786, 520],
            [150, 200],
        ]
    )
    aois = imp.Aois(arr, 5, 10, 7)
    new_aois = aois.remove_aoi_nearest_to_ref((0.5, 2))
    assert isinstance(new_aois, imp.Aois)
    assert new_aois.width == aois.width
    assert new_aois.frame == aois.frame
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.channel == aois.channel
    np.testing.assert_equal(new_aois.coords, arr[1:, :])


def test_save_load_aois():
    np.random.seed(0)
    arr = np.random.random_sample((20, 2)) * 512
    aois = imp.Aois(arr, 5, 10, 7)
    save_path = pathlib.Path(TEST_DATA_DIR / "save_path/aois")
    actual_path = save_path.with_suffix(".npz")
    if actual_path.exists() and actual_path.is_file():
        actual_path.unlink()
    aois.to_npz(save_path)
    assert actual_path.exists()
    assert actual_path.is_file()

    new_aois = imp.Aois.from_npz(actual_path)
    assert isinstance(new_aois, imp.Aois)
    assert isinstance(new_aois.width, int)
    assert new_aois.width == aois.width
    assert isinstance(new_aois.frame, int)
    assert new_aois.frame == aois.frame
    assert isinstance(new_aois.frame_avg, int)
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.channel == aois.channel
    np.testing.assert_equal(new_aois.coords, arr)


def test_get_aoi_intensity_centered_on_grid():
    height, width = 10, 30
    image = np.arange(width * height).reshape(height, width)
    aoi_width = 5
    assert aoi_width % 2  # AOI width must be odd for the following indexing to work
    coords = np.zeros((width * height, 2))
    for x in range(width):
        for y in range(height):
            coords[x + y * width, :] = [x, y]
    aois = imp.Aois(coords, 0, width=aoi_width)
    intensities = aois.get_intensity(image)
    for (x, y), intensity in zip(aois, intensities):
        x, y = int(x), int(y)
        if (
            y >= -0.5 + aoi_width / 2
            and x >= -0.5 + aoi_width / 2
            and x <= -0.5 + width - aoi_width / 2
            and y <= -0.5 + height - aoi_width / 2
        ):
            assert intensity == image[y, x] * aoi_width * aoi_width
        else:
            # Should return NaN if the AOI boundary get outside the image boundary
            assert np.isnan(intensity)


def test_get_aoi_intensity():
    rng = np.random.default_rng(0)
    height, width = 10, 30
    n_aois = 1000
    image = np.arange(width * height).reshape(height, width)
    for aoi_width in [5, 6]:  # odd and even aoi_width
        coords = np.stack(
            (
                rng.uniform(size=n_aois, high=width),
                rng.uniform(size=n_aois, high=height),
            )
        ).T
        for x in range(width):
            for y in range(height):
                coords[x + y * width, :] = [x, y]
        aois = imp.Aois(coords, 0, width=aoi_width)
        intensities = aois.get_intensity(image)
        for (x, y), intensity in zip(aois, intensities):
            if (
                y >= -0.5 + aoi_width / 2
                and x >= -0.5 + aoi_width / 2
                and x <= -0.5 + width - aoi_width / 2
                and y <= -0.5 + height - aoi_width / 2
            ):
                assert intensity == pytest.approx(
                    (y * width + x) * aoi_width**2, abs=1e-10, rel=1e-10
                )
            else:
                assert np.isnan(intensity), (
                    f"AOI at (x = {x}, y = {y}) with width {aoi_width} "
                    "is outside of the image boundary, but returned non-NaN intensity."
                )


def test_get_background_intensity():
    image = np.zeros((19, 19))
    image[7:12, 7:12] = 1
    image[np.logical_not(image)] = np.arange(336)
    aoi = imp.Aois(np.array([[9.1, 8.9]]), frame=0)
    bkg = aoi.get_background_intensity(image)
    assert bkg == np.median(np.arange(336)) * 25


def test_get_background_intensity_regression():
    rng = np.random.default_rng(0)
    width, height = 600, 400
    image = np.round(rng.uniform(0, 100, size=(height, width)))
    n_aois = 1000
    aoi_widths = [5, 6, 7, 8]
    max_width = max(aoi_widths)
    coords = np.stack(
        (
            rng.uniform(size=n_aois, low=max_width * 2, high=width - max_width * 2),
            rng.uniform(size=n_aois, low=max_width * 2, high=height - max_width * 2),
        )
    ).T
    with np.load(TEST_DATA_DIR / "backgrounds_regression.npz") as test_data:
        for aoi_width in aoi_widths:
            aois = imp.Aois(coords, width=aoi_width, frame=0)
            background = aois.get_background_intensity(image)
            np.testing.assert_allclose(background, test_data[str(aoi_width)])


def test_integration():
    data_file = sio.loadmat(TEST_DATA_DIR / "integration/test_integration.mat")
    aois = imp.Aois(data_file["coords"] - 1, frame=0)
    image = data_file["image"]
    true_intensity = data_file["intensity"]
    true_background = data_file["bkg"]
    intensity = np.fromiter(
        (aoi.get_intensity(image) for aoi in aois.iter_objects()), dtype=np.double
    )
    background = np.fromiter(
        (aoi.get_background_intensity(image) for aoi in aois.iter_objects()),
        dtype=np.double,
    )
    np.testing.assert_allclose(intensity, true_intensity.squeeze())
    np.testing.assert_allclose(background, true_background.squeeze())

    # Test output array
    intensity_2 = aois.get_intensity(image)
    np.testing.assert_equal(intensity_2, intensity)
    background_2 = aois.get_background_intensity(image)
    np.testing.assert_equal(background_2, background)


def test_colocalization_from_high_low_spots():
    interval_file = sio.loadmat(TEST_DATA_DIR / "20200228/L2_interval.dat")[
        "intervals"
    ]["green"].item()
    all_spots_high = interval_file["AllSpots"].item()["AllSpotsCells"].item()
    all_spots_low = interval_file["AllSpotsLow"].item()["AllSpotsCells"].item()
    atca = interval_file["AllTracesCellArray"].item()
    shifted_xy = sio.loadmat(TEST_DATA_DIR / "20200228/L2_shiftedxy.mat")["shiftedXY"]
    is_colocalized = np.zeros((850, shifted_xy.shape[0]))
    drifted_aois = [imp.Aois(shifted_xy[:, :, frame], frame) for frame in range(850)]
    ref_aoi_high = [
        imp.Aois(all_spots_high[frame, 0][:, :2], frame=frame) for frame in range(850)
    ]
    ref_aoi_low = (
        imp.Aois(all_spots_low[frame, 0][:, :2], frame=frame) for frame in range(850)
    )
    is_colocalized = imp._colocalization_from_high_low_spots(
        drifted_aois, ref_aoi_high, ref_aoi_low
    )
    for i_aoi in range(shifted_xy.shape[0]):
        np.testing.assert_equal(is_colocalized[:, i_aoi], atca[i_aoi, 0][:, 2])


def test_image_group():
    image_group = imp.ImageGroup(TEST_DATA_DIR / "20200228")
    assert isinstance(image_group.channels, tuple)
    assert isinstance(image_group.sequences, dict)
    assert len(image_group.channels) == len(image_group.sequences)
    for channel in image_group.channels:
        assert isinstance(channel, imp.Channel)
        assert isinstance(image_group.sequences[channel], imp.ImageSequence)


def test_image_group_iter():
    image_group = imp.ImageGroup(TEST_DATA_DIR / "20200228")
    for channel, sequence in image_group:
        assert isinstance(channel, imp.Channel)
        assert isinstance(sequence, imp.ImageSequence)


def test_read_glimpse_image_hdf5():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im_hdf5/"
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame in range(image_sequence.length):
        image = image_sequence.get_one_frame(i_frame)
        assert image.shape == (image_sequence.height, image_sequence.width)
        true_image = np.reshape(
            np.arange(i_frame + 1, i_frame + n_pixels + 1),
            (image_sequence.height, image_sequence.width),
        )
        np.testing.assert_equal(true_image, image)

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(-1)
        assert exception_info.value == "Frame number must be positive integers or 0"

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame("")
        assert exception_info.value == "Frame number must be positive integers or 0"

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(image_sequence.length)
        assert (
            exception_info.value
            == "Frame number ({}) exceeds sequence length - 1 ({})".format(
                image_sequence.length, image_sequence.length - 1
            )
        )


def test_image_sequence_class_hdf5():
    image_path = pathlib.Path(__file__).parent / "test_data/fake_im_hdf5/"
    image_sequence = imp.ImageSequence(image_path)
    assert image_sequence.width == 300
    assert image_sequence.height == 200
    assert image_sequence.length == 20
