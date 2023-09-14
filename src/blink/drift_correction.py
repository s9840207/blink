"""This module handles drift corrections."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.io as sio
import scipy.signal
from tqdm import tqdm

import blink.image_processing as imp


class DriftCorrector:
    def __init__(self,drifter):
        self._driftlist = drifter
        self._cum_driftlist = None

    def shift_aois(self, aois: "imp.Aois", target_frame: int) -> "imp.Aois":
        """
        Move the AOIs to another frame, adding the drift to the coordinates.

        The AOIs are constructed from the frame aois.frame. This function calculates
        the drift displacement from this frame to the target_frame, and return a new
        Aois object containing the coordinates with the added displacement.

        Args:
            aois: the Aois object to be shifted.
            target_frame: the target_frame the Aois to shift to

        Returns:
            the original Aois object if the DriftCorrector is empty, or a new
            Aois object with the shifted coordinates.
        """
        if self._driftlist is None:
            return aois
        start_frame = int(aois.frame)
        # Coerce because the drift list can terminate before the acquisition ends
        coerced_target_frame = min(target_frame, self._cum_driftlist.shape[0] - 1)
        if start_frame >= self._driftlist.shape[0]:  # Out of correction range
            drift = 0
        else:
            drift = (
                self._cum_driftlist[coerced_target_frame]
                - self._cum_driftlist[start_frame]
            )
        new_aois = imp.Aois(
            aois._coords + drift,
            frame=target_frame,
            width=aois.width,
            frame_avg=aois.frame_avg,
            channel=aois.channel,
        )
        return new_aois

    def _time_is_in_correction_range(self, time):
        return self._driftlist[:, 3].min() <= time <= self._driftlist[:, 3].max()

    def shift_aois_by_time(self, aois, time):
        if self._driftlist is None:
            return aois
        if not self._time_is_in_correction_range(time):
            pass
            # TODO: Probably need to add some check back
            # raise ValueError(f'time {time} is not in drift correction range ' +
            # f'[{self._driftlist[:, 3].min()}, {self._driftlist[:, 3].max()}]')
        # Assuming the frame number of Aois object is the same as driftlist
        # TODO: think a better way to do this
        f = scipy.interpolate.interp1d(
            self._cum_driftlist[:, 2],
            self._cum_driftlist[:, :2],
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=(self._cum_driftlist[0, :2], self._cum_driftlist[-1, :2]),
        )
        drift = f(time)
        new_aois = imp.Aois(
            aois._coords + drift,
            frame=aois.frame,  # TODO: need to fix this
            width=aois.width,
            frame_avg=aois.frame_avg,
            channel=aois.channel,
        )
        return new_aois

    @classmethod
    def from_imscroll(cls, path):
        driftlist = sio.loadmat(path)["driftlist"][
            :, [0, 2, 1]
        ]  # Swap x, y since imscroll image is transposed
        driftlist[:, 0] -= 1
        return cls._from_diff_driftlist(driftlist)

    def to_npy(self, path):
        np.save(path, self._driftlist, allow_pickle=False)

    @classmethod
    def from_npy(cls, path):
        driftlist = np.load(path, allow_pickle=False)
        return cls._from_diff_driftlist(driftlist)

    @classmethod
    def from_coords(
        cls, coords: np.ndarray, frame_range: Tuple[int, int], time=None, smooth=None
    ) -> "DriftCorrector":
        """
        Create DriftCorrector from coordinates time series of fiducial markers.

        Starting from the coordinates of fiducial markers in a certain frame range,
        this function will calculate the displacement of the marker relative to the
        first frame. The displacements of different fiducial markers are averaged.
        The resulting averaged displacement time serie is then soothed by the smooth
        method.
        The default is a Savitzky-Golay filter of order 2, with a window
        length of 11 frames.

        Args:
            coords: N x M x 2 array, the tracked coordinates of fiducial markers.
                    the first dimension corresponds to the frame numbers where the
                    coordinates were tracked from. The second dimension is for holding
                    multiple tracked coordinates sequences from fiducial markers. The
                    last dimension is for the x-y cooridinates.
            frame_range: a tuple marking the half-open range of the coords in the image
                         stack.
            time: required if wanting to use the shift_aois_by_time method of the
                  DriftCorrector.

        Returns:
            A DriftCorrector instance.
        """
        if coords.ndim != 3:
            raise ValueError(
                f"Invalid number of dimensions ({coords.ndim}) of coords, should be 3."
            )
        if coords.shape[2] != 2:
            raise ValueError(f"The length of a coordinate ({coords.shape[2]}) is not 2")
        start_frame, end_frame = frame_range
        if end_frame - start_frame != coords.shape[0]:
            raise ValueError(
                f"Mismatched length ({end_frame-start_frame}) of frame range "
                f"({start_frame}, {end_frame}) to the coordinates series length "
                f"({coords.shape[0]})"
            )
        diff_coords = np.diff(coords, axis=0, prepend=0)
        mean_diff_coords = np.mean(diff_coords, axis=1)
        mean_displacement = np.cumsum(mean_diff_coords, axis=0)
        if smooth is None:

            def smooth(x):
                return scipy.signal.savgol_filter(
                    x, axis=0, window_length=11, polyorder=2
                )

        filtered_displacement = smooth(mean_displacement)
        if time is None:
            driftlist = np.zeros((end_frame, 3))
        else:
            # TODO: add test for the with time case
            driftlist = np.zeros((end_frame, 4))
            driftlist[:, 3] = time
        driftlist[:, 0] = np.arange(end_frame)
        driftlist[start_frame + 1 : end_frame + 1, 1:3] = np.diff(
            filtered_displacement, axis=0
        )
        return cls._from_diff_driftlist(driftlist)

    @classmethod
    def _from_diff_driftlist(cls, driftlist):
        """
        Create a DriftCorrector from a Imscroll-style driftlist.

        The drift list contains three or four columns,
        depending whether the times of each frame are used. The columns are the frame
        number, the x-drift, the y-drift, and the time array.
        The driftlist[i, 1:3] corresponds to the shift of the coordinates between
        the (driftlist[i, 0]-1)th frame and the (driftlist[i, 0])th frame.
        """
        drifter = cls(drifter)
        drifter._driftlist = driftlist
        drifter._cum_driftlist = np.zeros((driftlist.shape[0], driftlist.shape[1] - 1))
        drifter._cum_driftlist[:, :2] = np.cumsum(driftlist[:, 1:3], axis=0)
        if driftlist.shape[1] == 4:
            drifter._cum_driftlist[:, 2] = driftlist[:, 3]
        return drifter


def drift_detection(image_sequence, frame_range, threshold, ref_aois=None):
    radius = 1.5
    aoi_list = []
    for frame, image in tqdm(
        zip(frame_range, image_sequence[frame_range]), total=len(frame_range)
    ):
        aoi_list.append(imp.pick_spots(image, threshold=threshold, frame=frame))
    
    if ref_aois is None:
        ref_aois = aoi_list[0]
    else:
        aoi_list[0] = ref_aois

    first_entry = True
    for i, aois in enumerate(aoi_list[1:], start=1):
        new_ref_aois = aois.remove_aois_far_from_ref(ref_aois, radius)
        if len(new_ref_aois) < 3 and first_entry:
            temp_frame_range = range(frame_range.start, frame_range.start + i)
            temp_aoi_list = aoi_list[:i]
            temp_ref_aois = ref_aois
            first_entry = False
        ref_aois = new_ref_aois
    
    if not len(ref_aois):
        frame_range = temp_frame_range
        aoi_list = temp_aoi_list
        ref_aois = temp_ref_aois

    last_aois = ref_aois
    drifted_coords = np.zeros((len(frame_range), len(ref_aois), 2))
    for i, aois, image in tqdm(
        (
            zip(
                reversed(range(len(frame_range))),
                reversed(aoi_list),
                image_sequence[reversed(frame_range)],
            )
        ),
        total=len(frame_range),
    ):
        ref_aois = aois.remove_aois_far_from_ref(ref_aois, radius)
        ref_aois.sort_by_ref(last_aois)
        coords = ref_aois.gaussian_refine(image).coords
        drifted_coords[i, :, :] = coords
    # breakpoint()

    plt.plot(
        np.arange(len(frame_range)),
        (
            drifted_coords[:, :, 0].squeeze() - drifted_coords[np.newaxis, 0, :, 0]
        ).squeeze(),
    )
    plt.show()
    plt.plot(
        np.arange(len(frame_range)),
        (
            drifted_coords[:, :, 1].squeeze() - drifted_coords[np.newaxis, 0, :, 1]
        ).squeeze(),
    )
    plt.show()

    time = image_sequence.time[frame_range]
    return DriftCorrector.from_coords(
        drifted_coords, (frame_range.start, frame_range.stop), time=time
    )
