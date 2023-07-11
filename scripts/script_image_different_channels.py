from pathlib import Path

import python_for_imscroll.image_processing as imp
import python_for_imscroll.script_colocalization_snapshot_image as tmp
import skimage.exposure
import skimage.io

import gui.image_view as iv


def main():
    image_group_dir = iv.select_directory_dialog()
    image_group = imp.ImageGroup(image_group_dir)
    frame = 0
    indices = {
        "blue": (slice(8, 268), slice(252, 512)),
        "green": (slice(8, 268), slice(252, 512)),
        "red": (slice(8, 268), slice(3, 263)),
    }

    for channel, sequence in image_group:
        image = sequence.get_one_frame(frame)
        image = image[indices[channel.em]]
        scale = tmp.quickMinMax(image)
        image = skimage.exposure.rescale_intensity(
            image, in_range=scale, out_range="uint8"
        )
        channel_name = "-".join(channel)
        save_path = image_group_dir / f"{channel_name}_frame{frame}.png"
        skimage.io.imsave(save_path, image)


if __name__ == "__main__":
    main()
