#  Copyright (C) 1997 John C. Crocker and David G. Grier (IDL version of spot picking
#  algorithm)
#  Copyright (C) 2004-2008 Daniel Blair and Eric Dufresne (MATLAB version of spot
#  picking algorithm)
#  Copyright (C) 2015 Larry Friedman, Brandeis University
#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (image_processing.py) is part of blink.
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

"""This module handles the processing to get data from image stacks."""

import itertools
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import scipy.interpolate
import scipy.io as sio
import scipy.ndimage
import scipy.optimize
from tqdm import tqdm

Channel = namedtuple("Channel", ["ex", "em"])


def is_glimpse_dir(path: Path):
    return (
        path.is_dir() and (path / "header.mat").exists and (path / "0.glimpse").exists
    )


class ImageGroup:
    """Represents a Glimpse multichannel video on disk.

    This class stores info about"""

    def __init__(self, image_group_dir: Path):
        image_sub_dirs = [
            path for path in image_group_dir.iterdir() if is_glimpse_dir(path)
        ]
        self.sequences = dict()
        self.channels = []
        for image_path in image_sub_dirs:
            image_sequence = ImageSequence(image_path)
            channel = image_sequence.get_channel()
            if channel in self.sequences:
                raise ValueError("Repeated channels in glimpse group")
            if channel.em == "G+R":
                channel1 = Channel(channel.ex, "green")
                channel2 = Channel(channel.ex, "red")
                self.sequences[channel1] = self.sequences[channel2] = image_sequence
                self.channels.append(channel1)
                self.channels.append(channel2)
                continue
            self.sequences[channel] = image_sequence
            self.channels.append(channel)
        self.channels = tuple(self.channels)
        
        ini_time = min([sequence.time[0] for sequence in self.sequences.values()])
        for channel, sequence in self.sequences.items():
            # TODO: Fix this selection issue. Since for G+R channel, the
            # sequences object for red and green emission is the same, but
            # listed under different key in the dict .sequences. Here, when
            # modifying the time, we will modify the same object twice. Simply
            # add a choice now, but not flexible.
            if channel == Channel("green", "red"):
                continue
            sequence.time -= ini_time

    def __iter__(self):
        return ((channel, self.sequences[channel]) for channel in self.channels)


class ImageSequence:
    """Represents an Glimpse image stack on disk.

    This class stores info about an Glimpse image stack on a disk and contains
    methods to read images as arrays from this image stack.

    Attributes:
        width: The width of the image stack in pixels.
        height: The height of the image stack in pixels.
        length: The number of images in the image stack.
    """

    def __init__(self, image_path: Path):
        self._image_path = image_path
        try:
            header_file = sio.loadmat(image_path / "header.mat")
            self._offset = header_file["vid"]["offset"][0, 0].squeeze()
            self._file_number = header_file["vid"]["filenumber"][0, 0].squeeze()
            self.width = header_file["vid"]["width"][0, 0].item()
            self.height = header_file["vid"]["height"][0, 0].item()
            self.length = header_file["vid"]["nframes"][0, 0].item()
            # TODO: need test for time attribute
            self.time = header_file["vid"]["ttb"][0, 0].squeeze() / 1000
        except NotImplementedError:
            with h5py.File(image_path / "header.mat", "r") as f:
                vid = f["vid"]
                self._offset = vid["offset"][:].astype("int").squeeze()
                self._file_number = vid["filenumber"][:].astype("int").squeeze()
                self.width = int(vid["width"][()])
                self.height = int(vid["height"][()])
                self.length = int(vid["nframes"][()])
                self.time = vid["ttb"][:].squeeze() / 1000

    def get_one_frame(self, frame: int) -> np.ndarray:
        """Read image of a specific frame from the sequence.

        Args:
            frame: The frame index of the requested image.

        Returns:
            The image array of the requested frame.
        """
        frame = int(frame)
        if frame < 0:
            raise ValueError("Frame number must be positive integers or 0.")
        if frame >= self.length:
            err_str = "Frame number ({}) exceeds sequence length - 1 ({})".format(
                frame, self.length - 1
            )
            raise ValueError(err_str)

        file_number = self._file_number[frame]
        offset = self._offset[frame]
        image_file_path = self._image_path / "{}.glimpse".format(file_number)
        image = np.fromfile(
            image_file_path,
            dtype=">i2",
            count=(self.width * self.height),
            offset=offset,
        )
        image = np.reshape(image, (self.height, self.width))
        # The glimpse saved image is U16 integer - 2**15 and saved in I16 format.
        # To recover the U16 integers, we need to add 2**15 back, but we cannot
        # do this directly since the image is read as I16 integer, adding 2**15
        # will cause overflow. Need to cast to larger container (at least 32 bit)
        # first.
        image = image.astype(int) + 2**15
        return image

    def __iter__(self):
        return (self.get_one_frame(frame) for frame in range(self.length))

    def get_averaged_image(self, start: int = 0, size: int = 1) -> np.ndarray:
        """Read images from frame range [start, start+size) and average them.

        Args:
            start: The starting index of the images to read from stack.
            size: The numbers of frames to average.

        Returns:
            The averaged image array.
        """
        start = int(start)
        size = int(size)
        image = 0
        for frame in range(start, start + size):
            image += self.get_one_frame(frame)
        return image / size

    def get_channel(self):
        laser_dict = {
            "Blue 488 nm": "blue",
            "Green 532 nm": "green",
            "Red 635 nm": "red",
        }
        filter_dict = {
            "red": "red",
            "G572/15": "green",
            "3": "ir",
            "B517/20": "blue",
            "G+R": "G+R",
        }
        try:
            header_file = sio.loadmat(self._image_path / "header.mat")["vid"]
            lasers = header_file["lasers"][0, 0][0, :]
            if np.count_nonzero(lasers) != 1:
                raise ValueError("More than one laser is on")
            excitation = (
                header_file["laser_names"][0, 0][np.nonzero(lasers)].item().strip()
            )
            emission = (
                header_file["filter_names"][0, 0][header_file["filters"][0, 0][0, 0]]
                .item()
                .strip()
            )
            return Channel(ex=laser_dict[excitation], em=filter_dict[emission])
        except NotImplementedError:
            with h5py.File(self._image_path / "header.mat", "r") as f:
                vid = f["vid"]
                lasers = vid["lasers"][:, 0].astype(int)
                if np.count_nonzero(lasers) != 1:
                    raise ValueError("More than one laser is on")
                laser_names_array = vid.get("laser_names")[:].T  # Transpose required
                filter_names_array = vid.get("filter_names")[:].T
                # Need to remove ascii character \x00
                laser_names = np.array(
                    [
                        "".join(chr(i) for i in laser_names_array[j]).strip("\x00")
                        for j in range(4)
                    ]
                )
                filter_names = np.array(
                    [
                        "".join(chr(i) for i in filter_names_array[j])
                        .strip("\x00")
                        .strip()
                        for j in range(8)
                    ]
                )
                excitation = laser_names[np.nonzero(lasers)]
                emission = filter_names[vid["filters"][0].astype(int).squeeze()]
                return Channel(ex=laser_dict[excitation[0]], em=filter_dict[emission])

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = range(self.length)[key]
        try:
            return (self.get_one_frame(frame) for frame in key)
        except TypeError:
            # If key is not iterable
            return (self.get_one_frame(frame) for frame in [key])


def conv2(v1, v2, mat, mode="same"):
    """
    Two-dimensional convolution of matrix mat by vectors v1 and v2

    This function mimicks the behavior of MATLAB conv2.

    First convolves each column of 'mat' with the vector 'v1'
    and then it convolves each row of the result with the vector 'v2'.

    See stack overflow questions 24231285 Is there a python equivalent to MATLAB's conv2

    Args:
        v1: Input vector. Convolves with each column of 'mat'.
        v2: Second input vector. Convolves with the convolution of 'v1' with the
            columns of 'mat'.
        mat: A 1D or 2D array that will be convolved.
        mode: See the doc of numpy.convolve

    Returns:
        convolved_mat: The convolved array.
    """
    tmp = np.apply_along_axis(np.convolve, 0, mat, v1, mode)
    convolved_mat = np.apply_along_axis(np.convolve, 1, tmp, v2, mode)
    return convolved_mat


def _normalize(x):
    return x / np.sum(x)


def band_pass(image: np.ndarray, r_noise: float, r_object: int) -> np.ndarray:
    """
    Performs a bandpass filter on the input image.

    To suppress pixel noise and long-wavelength image variations while retaining
    information of a characteristic size.

    Performs a bandpass by a two part process. First, a lowpassed image is
    produced by convolving the original with a gaussian. Next, a second
    lowpassed image is produced by convolving the original with a boxcar
    function. By subtracting the boxcar version from the gaussian version, we
    are using the boxcar version to perform a highpass.

    Originally is the MATLAB bpass function written by David G. Grier and
    John C. Crocker. See http://site.physics.georgetown.edu/matlab/.

    Args:
        image: The two-dimensional array to be filtered.
        r_noise: Characteristic lengthscale of noise in pixels. Additive noise
                 averaged over this length should vanish. May assume any
                 positive floating value. When set to 0, only the highpass
                 "background subtraction" operation is performed.

        r_object: Integer length in pixels somewhat larger than a typical object.
                  Can also be set to 0, in which case only the lowpass "blurring"
                  operation defined by 'r_noise' is done, without the background
                  subtraction defined by 'r_object'.

    Returns:
        filtered image.
    """

    if r_noise:
        gauss_kernel_size = np.ceil(r_noise * 5)
        x = np.arange(-gauss_kernel_size, gauss_kernel_size + 1)
        gaussian_kernel = _normalize(np.exp(-((x / r_noise / 2) ** 2)))
    else:
        gaussian_kernel = 1
    gauss_filtered_image = conv2(gaussian_kernel, gaussian_kernel, image)

    if r_object:
        boxcar_kernel = _normalize(np.ones(round(r_object) * 2 + 1))
        boxcar_image = conv2(boxcar_kernel, boxcar_kernel, image)
        band_passed_image = gauss_filtered_image - boxcar_image
    else:
        band_passed_image = gauss_filtered_image

    edge_size = round(max(r_object, gauss_kernel_size))
    band_passed_image[0:edge_size, :] = 0
    band_passed_image[-edge_size:, :] = 0
    band_passed_image[:, 0:edge_size] = 0
    band_passed_image[:, -edge_size:] = 0
    band_passed_image[band_passed_image < 0] = 0
    return band_passed_image


def find_peaks(image: np.ndarray, threshold: float, peak_size: float) -> np.ndarray:
    """
    Finds local maxima in an image to pixel level accuracy.

    This provides a rough guess of particle centers to be used by
    localize_centroid().

    Translated from pkfnd.m written in MATLAB.
    See http://site.physics.georgetown.edu/matlab/.

    Notes:
        Particle should be bright spots on dark background with little noise.
        Often an bandpass filtered brightfield image or a nice fluorescent image.

    Args:
        image: The 2D array to process.
        threshold: The minimum brightness of a pixel that might be local maxima.
        peak_size: The expected size of a peak. If closed peaks in radius of
                   peak_size/2 were found, they will be presented by only the
                   brightest. Also removes all peaks withing peak_size from the
                   image boundary.
                   If image noisy such that a single particle has multiple local
                   maxima), then set peak_size to a value slightly larger than
                   the diameter of your particle.

    Returns:
        A N x 2 array with each row containing (x, y) coordinates of a local
        maximum on the image.
        Returns None when no local maximum was found.
    """
    # Transpose the image to mimick the matlab index ordering, since the later
    # close peak removal is dependent on the order of the peaks.
    t_image = image.T
    is_above_th = t_image > threshold
    if not is_above_th.any():
        print("nothing above threshold")
        return None
    width, height = t_image.shape

    is_local_max = scipy.ndimage.maximum_filter(t_image, size=3) == t_image
    idx = np.logical_and(is_above_th, is_local_max)  # np.nonzero returns the indices
    # The index is labeled as x, y since the image was transposed
    x_idx, y_idx = idx.nonzero()

    is_not_at_edge = np.logical_and.reduce(
        (
            x_idx > peak_size - 1,
            x_idx < width - peak_size - 1,
            y_idx > peak_size - 1,
            y_idx < height - peak_size - 1,
        )
    )

    y_idx = y_idx[is_not_at_edge]
    x_idx = x_idx[is_not_at_edge]

    if len(y_idx) > 1:
        c = peak_size // 2
        # Create an image with only peaks
        peak_image = np.zeros(t_image.shape)
        peak_image[x_idx, y_idx] = t_image[x_idx, y_idx]
        for x, y in zip(y_idx, x_idx):
            roi = peak_image[y - c : y + c + 2, x - c : x + c + 2]
            max_peak_pos = np.unravel_index(np.argmax(roi), roi.shape)
            max_peak_val = roi[max_peak_pos[0], max_peak_pos[1]]
            peak_image[y - c : y + c + 2, x - c : x + c + 2] = 0
            peak_image[y - c + max_peak_pos[0], x - c + max_peak_pos[1]] = max_peak_val
        x_idx, y_idx = (peak_image > 0).nonzero()
    peaks = np.stack((x_idx, y_idx), axis=-1)
    return peaks


def localize_centroid(image: np.ndarray, peaks: np.ndarray, dia: int) -> np.ndarray:
    """
    Calculates the centroid of peaks on an image to sub-pixel accuracy.

    Translated from cntrd.m written by Eric R. Dufresne in MATLAB.
    See http://site.physics.georgetown.edu/matlab/.

    Notes:
        - Particle should be bright spots on dark background with little noise.
          Often an bandpass filtered brightfield image or a nice fluorescent
          image.
        - If find_peaks, and localize_centroid return more then one location per
          particle, then you should try to filter the input image more carefully.
          If you still get more than one peak for a particle, use the peak_size
          parameter in find_peaks.
        - If you want sub-pixel accuracy, you need to have a lot of pixels in
          your window (dia>>1). To check for pixel bias, plot a histogram of the
          fractional parts of the resulting locations.

    Args:
        image: The 2D array to process.
        peaks: A M x 2 integer array with each row containing (x, y) coordinates
               of a peak on the image.
        dia: Odd integer. The diamter of the window over which to average to
             calculate the centroid. This value should be big enough to capture
             the whole particle but not so big that it captures others.
             Recommended size is the long lengthscale used in band_pass() plus 2.

    Returns:
        peaks_out. A N x 2 float array with each row containing (x, y)
                   coordinates to sub-pixel accuracy of a peak on the image.
    """
    if peaks is None:  # There is no peak found by find_peaks()
        return None
    if dia % 2 != 1:
        raise ValueError("Window diameter only accepts odd integer values.")
    if peaks.size == 0:
        raise ValueError("There are no peaks input")
    # Filter out the peaks too close to the edges
    height, width = image.shape
    x_idx = peaks[:, 0]
    y_idx = peaks[:, 1]
    is_in_range = np.logical_and.reduce(
        (
            x_idx > 1.5 * dia,
            x_idx < width - 1.5 * dia,
            y_idx > 1.5 * dia,
            y_idx < height - 1.5 * dia,
        )
    )
    peaks = peaks[is_in_range, :]

    radius = int((dia + 1) / 2)
    x_weight = np.tile(np.arange(2 * radius), (2 * radius, 1))
    y_weight = x_weight.T
    mask = _create_circular_mask(2 * radius, radius=radius)

    peaks_out = np.zeros(peaks.shape)
    for i, (x, y) in enumerate(peaks):
        masked_roi = (
            mask
            * image[y - radius + 1 : y + radius + 1, x - radius + 1 : x + radius + 1]
        )
        norm = np.sum(masked_roi)
        x_avg = np.sum(masked_roi * x_weight) / norm + (x - radius + 1)
        y_avg = np.sum(masked_roi * y_weight) / norm + (y - radius + 1)
        peaks_out[i, :] = [x_avg, y_avg]
    return peaks_out


def _create_circular_mask(w, center=None, radius: float = None):
    if center is None:
        # Use the middle of the image, which is the average of 1st index 0 and
        # last index w-1
        center = ((w - 1) / 2, (w - 1) / 2)
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], w - center[1])

    y_grid, x_grid = np.ogrid[:w, :w]
    # Use broadcasting to calculate the distaces of each element
    dist_from_center = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def pick_spots(
    image, threshold=50, noise_dia=1, spot_dia=5, frame=0, aoi_width=5, frame_avg=1
) -> "Aois":
    """
    Pick spots on an image and outputs the spots to an Aois object.

    Args:
        image: The 2D array to pick spot from.
        threshold: The brightness threshold on the band-passed image that local
                   maxima with brightness above which will be classified as
                   spots.
        noise_dia: The characteristic size of the noise on the image. Will be
                   used in the band pass filter.
        spot_dia: Odd integer. Should be slightly larger than the estimation of
                  the spot diameter.
        frame: The frame index of the image in the image stack.
        aoi_width: The size of AOI to construct the Aois object.
        frame_avg: The frame average of the image.

    Returns:
        The Aois object containing the localized centroid of spots.
    """
    filtered_image = band_pass(image, r_noise=noise_dia, r_object=spot_dia)
    peaks = find_peaks(filtered_image, threshold=threshold, peak_size=spot_dia)
    peak_centroids = localize_centroid(filtered_image, peaks=peaks, dia=spot_dia)
    return Aois(peak_centroids, frame=frame, width=aoi_width, frame_avg=frame_avg)


class Aois:
    """Stores the coordinates info of AOIs and contains image indexing methods.

    Attributes:
        width: The width of the AOI in pixels.
    """

    def __init__(
        self,
        coords: np.ndarray,
        frame: int,
        frame_avg: int = 1,
        width: int = 5,
        channel=None,
    ):
        self._frame_avg = frame_avg
        self._frame = frame
        self.width = width
        self.coords = coords
        self._channel = channel

    def get_all_x(self):
        return self.coords[:, 0]

    def get_all_y(self):
        return self.coords[:, 1]

    @property
    def channel(self) -> str:
        """The image channel where these AOIs are picked."""
        return self._channel

    @channel.setter
    def channel(self, value: str):
        if self._channel is None:
            self._channel = value
        else:
            raise AttributeError("can't overwrite defined channel")

    @property
    def frame(self):
        """The image frame index where these AOIs are picked"""
        return self._frame

    @property
    def frame_avg(self):
        """The image frame average where these AOIs are picked"""
        return self._frame_avg

    def __len__(self):
        if len(self.coords.shape) == 1:
            return 1
        return self.coords.shape[0]

    def __iter__(self):
        return map(tuple, self.coords)

    def __contains__(self, item):
        if len(item) == 2:
            in_coord = np.array(item)
            return (self.coords == in_coord).all(axis=1).any()
        return False

    def remove_close_aois(self, distance: int = 0) -> "Aois":
        """Find AOIs with center distance <= 'distance' and return an Aois object
        without them.

        Any AOI within the radius range of 'distance' from any other AOI in this
        object will be marked as 'is_close'. This means more than 2 AOIs forming
        a cluster will also be removed.

        Args:
            distance: The radius threshold to classify if two AOIs are close to
                      each other.

        Returns:
            new_aois. A new Aois object without the 'is_close' AOIs.
        """
        x = self.get_all_x()[np.newaxis]
        y = self.get_all_y()[np.newaxis]
        x_diff_squared = (x - x.T) ** 2
        y_diff_squared = (y - y.T) ** 2
        dist_arr = np.sqrt(x_diff_squared + y_diff_squared)
        # The diagonal is the diff of the same AOI, which will always == 0.
        # Use masking to ignore.
        is_diag = np.identity(len(self), dtype=bool)
        is_not_close = np.logical_or(dist_arr > distance, is_diag).all(axis=1)
        new_aois = Aois(
            self.coords[is_not_close, :],
            frame=self.frame,
            frame_avg=self.frame_avg,
            width=self.width,
            channel=self.channel,
        )
        return new_aois

    def is_in_range_of(self, ref_aois: "Aois", radius: Union[int, float]) -> np.ndarray:
        dist_arr = self._dist_arr(ref_aois)
        return (dist_arr <= radius).any(axis=1)

    def remove_aois_near_ref(
        self, ref_aois: "Aois", radius: Union[int, float]
    ) -> "Aois":
        is_in_range_of_ref = self.is_in_range_of(ref_aois=ref_aois, radius=radius)
        new_aois = Aois(
            self.coords[np.logical_not(is_in_range_of_ref), :],
            frame=self.frame,
            frame_avg=self.frame_avg,
            width=self.width,
            channel=self.channel,
        )
        return new_aois

    def remove_aois_far_from_ref(
        self, ref_aois: "Aois", radius: Union[int, float]
    ) -> "Aois":
        is_in_range_of_ref = self.is_in_range_of(ref_aois=ref_aois, radius=radius)
        new_aois = Aois(
            self.coords[is_in_range_of_ref, :],
            frame=self.frame,
            frame_avg=self.frame_avg,
            width=self.width,
            channel=self.channel,
        )
        return new_aois

    def _get_params(self):
        return {"frame": self.frame, "frame_avg": self.frame_avg, "width": self.width}

    def iter_objects(self):
        gen_objects = (
            Aois(self.coords[i, np.newaxis, :], **self._get_params())
            for i in range(len(self))
        )
        return gen_objects

    def get_subimage_slice(self, width: Optional[int] = None) -> Tuple[slice, slice]:
        """
        Return the (row, column) slices of an AOI to select the corresponding subimage.

        This function only works with single AOI, which could be generated by the
        iter_objects() method.
        The nearest AOI box is selected according to its center and the (optional)
        specified width. If the AOI box crosses over the top and the left edges of
        the image, the subimage slice will be truncated automatically to avoid
        indexing error.

        Args:
            width: the width of the AOI subimage, will use the value from the
            Aois object if unspecified.

        Returns:
            A tuple of two slices, which could be used to slice the image in the
            row (y) and column (x) order.
        """
        if len(self) != 1:
            raise ValueError(f"Wrong AOI length ({len(self)}), must be 1.")
        if width is None:
            width = self.width
        width = int(width)  # The slice only accepts int otherwise will TypeError
        # Center minus half-width will be the edge of AOI, add 0.5 to be the first
        # Grid point
        origin_float = self.coords[0] - width / 2 + 0.5
        origin_x = int(round(origin_float[0].item()))
        origin_y = int(round(origin_float[1].item()))
        return (
            slice(max(0, origin_y), origin_y + width),
            slice(max(0, origin_x), origin_x + width),
        )

    def gaussian_refine(self, image: np.ndarray) -> "Aois":
        """
        Refine the AOIs coordinates using 2-D Gaussian fitting on a given image.

        Starting with the coordinates in the image, sub-images with the size of the
        width attribute of this Aois object are constructed. Then, 2-D Gaussian
        fitting is used to obtain finer peak centers (if it exists), and a new Aois
        object is returned containing the new coordinates.

        Args:
            image: 2-D ndarray, the image to perform 2-D Gaussian fitting on.

        Returns:
            The new Aois object storing the refined coordinates. Other attributes
            will be copied from self.
        """
        fit_result = np.zeros((len(self), 5))

        for i, aoi in enumerate(self.iter_objects()):
            subimg_slice = aoi.get_subimage_slice()
            height, width = [
                slice_obj.stop - slice_obj.start for slice_obj in subimg_slice
            ]
            subimg_origin = [slice_obj.start for slice_obj in subimg_slice]
            subimg_origin.reverse()
            x = np.arange(width)[np.newaxis, :]
            y = np.arange(height)[np.newaxis, :]
            xy = [x, y]
            z = image[subimg_slice].ravel()
            fit_result[i] = fit_2d_gaussian(xy, z)
            fit_result[i, 1:3] += subimg_origin

        new_aois = Aois(
            fit_result[:, 1:3],
            frame=self.frame,
            frame_avg=self.frame_avg,
            width=self.width,
            channel=self.channel,
        )
        return new_aois

    def __add__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            new_aois = Aois(
                np.concatenate((self.coords, np.array(other)[np.newaxis]), axis=0),
                frame=self.frame,
                frame_avg=self.frame_avg,
                width=self.width,
                channel=self.channel,
            )
            return new_aois
        raise TypeError("Aois class addition only accepts tuples with len == 2")

    def remove_aoi_nearest_to_ref(self, ref_aoi: Tuple[float, float]) -> "Aois":
        ref_aoi = np.array(ref_aoi)[np.newaxis]
        dist = np.sum((self.coords - ref_aoi) ** 2, axis=1)
        removing_idx = np.argmin(dist)
        new_coords = np.delete(self.coords, removing_idx, axis=0)
        new_aois = Aois(
            new_coords,
            frame=self.frame,
            frame_avg=self.frame_avg,
            width=self.width,
            channel=self.channel,
        )
        return new_aois

    def to_npz(self, path):
        params = {"frame": self.frame, "width": self.width, "frame_avg": self.frame_avg}
        names = ["key", "value"]
        formats = ["U10", int]
        dtype = dict(names=names, formats=formats)
        params = np.fromiter(params.items(), dtype=dtype, count=len(params))
        np.savez(
            path,
            params=params,
            coords=self.coords,
            channel=np.array(self.channel, dtype="U5"),
        )

    def to_imscroll_aoiinfo2(self, path):
        aoiinfo = np.zeros((len(self), 6))
        aoiinfo[:, 0] = self.frame
        aoiinfo[:, 1] = self.frame_avg
        aoiinfo[:, 2] = self.get_all_y() + 1
        aoiinfo[:, 3] = self.get_all_x() + 1
        aoiinfo[:, 4] = self.width
        aoiinfo[:, 5] = np.arange(1, len(self) + 1)
        sio.savemat(path.with_suffix(".dat"), dict(aoiinfo2=aoiinfo))

    @classmethod
    def from_imscroll_aoiinfo2(cls, file):
        """
        Load an Aois object from a MAT file containing the Imscroll aoiinfo2 matrix.

        Args:
            file: file-like object, string, or pathlib.Path.
                  The file to read. Will be passed to scipy.io.loadmat.

        Returns:
            An Aois instance that holds the data from the MAT file.
        """
        aoiinfo = sio.loadmat(file)["aoiinfo2"]
        aois = cls(
            aoiinfo[:, [3, 2]] - 1,
            frame=int(aoiinfo[0, 0] - 1),
            frame_avg=int(aoiinfo[0, 1]),
            width=int(aoiinfo[0, 4]),
        )
        return aois

    @classmethod
    def from_npz(cls, file):
        """
        Load an Aois object from an npz file generated by the save_npz method.

        Args:
            file: file-like object, string, or pathlib.Path.
                  The file to read. Will be passed to numpy.load.

        Returns:
            An Aois instance that holds the data from the npz file.
        """
        npz_file = np.load(file, allow_pickle=False)
        channel = str(npz_file["channel"])
        if channel == "None":
            channel = None
        params_arr = npz_file["params"]
        params = {row["key"]: int(row["value"]) for row in params_arr}
        coords = npz_file["coords"]
        aois = cls(coords, channel=channel, **params)
        return aois

    def get_interp2d_grid(self):
        offset = self.width / 2 - 0.5
        # Note: cannot use arange directly with floats, as sometimes
        # floating point error will create array with length > self.width
        arange_width = np.arange(self.width)

        # Need this because the self.coords is now 1D, cannot directly loop
        # over it
        if len(self) == 1:
            coords = self.coords[np.newaxis, :]
        else:
            coords = self.coords

        grid_list = []
        for xy in coords:
            xy = xy.squeeze()  # reduce the dimension so it can be unpacked
            x_start, y_start = xy - offset
            grid_list.append((x_start + arange_width, y_start + arange_width))
        return tuple(grid_list)

    def get_intensity(self, image):
        y_max, x_max = image.shape
        f = scipy.interpolate.interp2d(
            *np.ogrid[:x_max, :y_max], image, fill_value=np.nan
        )
        intensities = (np.sum(f(*grid)) for grid in self.get_interp2d_grid())
        intensities = np.fromiter(intensities, dtype=np.double)
        return intensities

    def get_background_intensity(self, image):
        backgrounds = np.zeros(len(self), dtype=np.double)
        for i, aoi in enumerate(self.iter_objects()):
            sub_im_slice = aoi.get_subimage_slice(width=2 * self.width + 9)
            sub_im = image[sub_im_slice]
            origin = [slice_obj.start for slice_obj in sub_im_slice]
            aoi_slice = aoi.get_subimage_slice(width=2 * self.width + 1)
            aoi_ogrid = np.ogrid[aoi_slice]
            mask = np.ones(sub_im.shape, dtype=bool)
            try:
                mask[aoi_ogrid[0] - origin[0], aoi_ogrid[1] - origin[1]] = 0
            except:
                print(i, aoi.coords)
            backgrounds[i] = np.median(sub_im[mask]) * self.width**2
        return backgrounds

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        if coords is None:
            # TODO: The None type is left here to preserve behavior, need to
            # remove later.
            self._coords = None
        elif np.any(coords < 0):
            raise ValueError("Coordinates of an AOI should be positive valued or 0.")
        elif coords.ndim != 2:
            raise ValueError(
                f"The ndim of the coordinates array should be 2. "
                f"Instead we have {coords.ndim}."
            )
        else:
            self._coords = coords

    def _dist_arr(self, ref_aois: "Aois"):
        x = self.get_all_x()[:, np.newaxis]
        y = self.get_all_y()[:, np.newaxis]
        ref_x = ref_aois.get_all_x()[np.newaxis]  # Produce row vector
        ref_y = ref_aois.get_all_y()[np.newaxis]
        dist_arr = np.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2)
        return dist_arr

    def sort_by_ref(self, ref_aois: "Aois"):
        dist_arr = self._dist_arr(ref_aois)
        index = np.argmin(dist_arr, axis=0)
        for i, j in enumerate(index):
            if dist_arr[i, j] > 5:
                Warning("Aois misaligned")
        self.coords = self.coords[index, :]

    def __getitem__(self, index):
        try:
            new_aoi = Aois(self.coords[index, np.newaxis, :], **self._get_params())
        except IndexError:
            raise IndexError(
                f"index 10 is out of bounds for Aois with length {len(self)}"
            )
        return new_aoi


def symmetric_2d_gaussian(xy, A, x0, y0, sigma, h):
    x, y = xy
    y = y.T
    denominator = (x - x0) ** 2 + (y - y0) ** 2
    return (A * np.exp(-denominator / (2 * sigma**2)) + h).ravel()


def fit_2d_gaussian(xy, z):
    x_c = xy[0].mean()
    y_c = xy[1].mean()
    bounds = (0, [2 * z.max(), xy[0].max(), xy[1].max(), 10000, 2 * z.max()])
    param_0 = [z.max() - z.mean(), x_c, y_c, min(x_c, y_c) / 2, z.mean()]
    popt, _ = scipy.optimize.curve_fit(
        symmetric_2d_gaussian,
        xy,
        z,
        p0=param_0,
        bounds=bounds,
        ftol=1e-6,
        gtol=1e-6,
        xtol=1e-10,
        max_nfev=10000,
    )
    return popt


def analyze_colocalization(aois, image_sequence, thresholds, drift_corrector=None):
    is_colocalized = np.zeros((image_sequence.length, len(aois)), dtype=bool)
    ref_aoi_high = []
    ref_aoi_low = []
    for image in tqdm(image_sequence, total=image_sequence.length):
        ref_aoi_high.append(pick_spots(image, threshold=thresholds[0]))
        ref_aoi_low.append(pick_spots(image, threshold=thresholds[1]))
    if drift_corrector is not None:
        aois = [
            drift_corrector.shift_aois(aois, frame)
            for frame in range(image_sequence.length)
        ]
    is_colocalized = _colocalization_from_high_low_spots(
        aois, ref_aoi_high, ref_aoi_low
    )
    return is_colocalized


def _colocalization_from_high_low_spots(
    aois: "Aois", ref_high_aois_list, ref_low_aois_list
):
    n_frames = len(ref_high_aois_list)
    if isinstance(aois, Aois):
        n_aois = len(aois)
        aois = itertools.repeat(aois, n_frames)
    else:
        n_aois = len(aois[0])
    is_colocalized = np.zeros((n_frames, n_aois), dtype=bool)
    for frame, (current_aois, ref_aoi_high, ref_aoi_low) in enumerate(
        zip(aois, ref_high_aois_list, ref_low_aois_list)
    ):
        in_range_high = current_aois.is_in_range_of(ref_aoi_high, 1.5)
        in_range_low = current_aois.is_in_range_of(ref_aoi_low, 1.5 * 1.5)
        is_colocalized[frame, :] = np.logical_or(
            in_range_high, np.logical_and(in_range_low, is_colocalized[frame - 1, :])
        )
    return is_colocalized
