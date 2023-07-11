import argparse
from pathlib import Path

import blink.mapping
import gui.image_view
import trace_plotter.trace_plotter


def view_image(args):
    gui.image_view.main()


def plot_traces(args):
    trace_plotter.trace_plotter.main()


def make_mapping_file(args):
    source_dir = args.source_dir.expanduser().resolve()
    output = args.output.expanduser().resolve().with_suffix(".npz")
    blink.mapping.make_mapping_from_directory_of_spots(
        source_dir, output, args.d, args.d2
    )


def entry():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_view_image = subparsers.add_parser(
        "view-image", help="opens the image view GUI"
    )
    parser_view_image.set_defaults(func=view_image)
    parser_plot_traces = subparsers.add_parser(
        "plot-traces", help="opens the trace plotter GUI"
    )
    parser_plot_traces.set_defaults(func=plot_traces)
    parser_make_map = subparsers.add_parser(
        "make-mapping-file",
        help="Makes a mapping file from a set of AOIs files by finding matching pairs.",
        description=(
            "Makes a mapping file from a set of AOIs files by finding matching pairs. "
            "The source_dir contains only the AOIs files. "
            "The AOIs files of the same field of view should have the same base name, "
            "and a channel string ('_r', '_g', '_b') should be appended at the end of "
            "the base name to specify channels. "
            "For example, my source_dir has three files ['0126_0_r.npz', '0126_0_g.npz'"
            ", '0126_0_b.npz']."
        ),
    )
    parser_make_map.add_argument(
        "source_dir",
        help="The path to the directory that contains the AOIs files.",
        type=Path,
    )
    parser_make_map.add_argument(
        "-o",
        "--output",
        help=(
            "The path to save the mapping file. "
            "A '.npz' suffix will be appended/changed to automatically. "
            "If omitted, '~/map.npz' will be used"
        ),
        type=Path,
        default=Path("~/map.npz"),
    )
    parser_make_map.set_defaults(func=make_mapping_file)
    parser_make_map.add_argument(
        "-d",
        help=("The distance threshold to find colocalized spots. " "Defaults to 3."),
        type=float,
        default=3,
    )
    parser_make_map.add_argument(
        "-d2",
        help=(
            "The distance threshold to refine the final mapping. "
            "Any pair with mapping error larger than this value will be iteratively "
            "removed. "
            "Defaults to 0.6."
        ),
        type=float,
        default=0.6,
    )

    # parse the args and call whatever function was selected
    args = parser.parse_args()
    args.func(args)
