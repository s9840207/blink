from pathlib import Path

import numpy as np
import python_for_imscroll.image_processing as imp
import scipy.io as sio
import skimage.io
from python_for_imscroll import binding_kinetics
from skimage import exposure


def read_coordinate_sequences(int_corrected_path, channel):
    file = sio.loadmat(int_corrected_path)
    aoifits_array = file["aoifits"]["data" + channel][0, 0]
    coords = aoifits_array[:, [3, 4]]
    n_frames = int(max(aoifits_array[:, 1]))
    n_aoi = int(max(aoifits_array[:, 0]))
    coords = np.reshape(coords, (n_frames, n_aoi, 2))
    coords = np.swapaxes(coords, 1, 2)
    return coords


def main():
    aoi = 7
    datapath = Path(
        "/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200317/20200317imscroll/"
    )
    image_path = Path(
        "/run/media/tzu-yu/data/PriA_project/Expt_data/20200317/L5_GstPriA_125pM/L5_02_photobleaching_03/hwligroup00821/"
    )
    image_sequence = imp.ImageSequence(image_path)
    filestr = "L5_02_03"
    framestart = 0
    int_corrected_path = datapath / (filestr + "_intcorrected.dat")
    try:
        all_data, AOI_categories = binding_kinetics.load_all_data(
            datapath / (filestr + "_all.json")
        )
    except FileNotFoundError:
        print("{} file not found".format(filestr))
    channel = "green"
    channel_data = all_data["data"].sel(channel=channel)
    green_coord = read_coordinate_sequences(int_corrected_path, channel)
    green_coord_aoi = green_coord[:, :, aoi - 1]
    interval = all_data["intervals"]
    aoi_interval = interval.sel(AOI=aoi)
    aoi_interval = aoi_interval.dropna(dim="interval_number")

    state_sequence = channel_data["viterbi_path"].sel(state="label", AOI=aoi)
    state_start_index = binding_kinetics.find_state_end_point(state_sequence)
    if state_start_index.any():
        for event_end in state_start_index:
            spacer = 1
            out = np.zeros((11, 11 * 6 + 5 * spacer), dtype="uint16") - 1
            spacer_list = []
            a = np.arange(spacer)
            for i in range(5):
                a += 11
                spacer_list.extend(a.tolist())
                a += spacer

            out[:, spacer_list] = 150 * 2**8
            for i, frame in enumerate(range(event_end - 2, event_end + 4)):
                coord = np.round(green_coord_aoi[frame, :]) - 1
                img = image_sequence.get_one_frame(frame + framestart)
                scale = (500, 3000)
                dia = 11
                x = int(coord[1])
                y = int(coord[0])
                rad = int((dia - 1) / 2)
                sub_img = img[y - rad : y + rad + 1, x - rad : x + rad + 1]

                im = exposure.rescale_intensity(
                    sub_img, in_range=scale, out_range=(2**13, 2**16 - 1)
                )
                arr = np.zeros(sub_img.shape, dtype="uint8")
                arr[:, :] = im
                out[:, i * (11 + spacer) : i * (11 + spacer) + 11] = im
            save_path = datapath / "{}_{}_{:.0f}.png".format(
                filestr, aoi, channel_data.time[event_end].item() + 24.226
            )
            out = skimage.util.invert(out)
            skimage.io.imsave(save_path, out)


if __name__ == "__main__":
    main()
