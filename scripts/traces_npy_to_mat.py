from pathlib import Path

import time_series


def main():
    datadir = Path(
        r"D:\CWH\20221222\1"
    )
    for path in datadir.iterdir():
        if path.stem[-7:] == "_traces":
            filestr = path.stem[:-7]
            npz_path = datadir / f"{filestr}_traces.npz"
            traces = time_series.TimeTraces.from_npz(npz_path)
            traces.to_mat(npz_path.with_suffix(".dat"))


if __name__ == "__main__":
    main()
