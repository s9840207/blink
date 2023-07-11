from pathlib import Path

import image_processing as imp
import matplotlib.pyplot as plt
import numpy as np
import time_series as ts


def main():
    gamma = 0.31
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    data_dir = Path(
        "/mnt/data/Research/PriA_project/analysis_result/20211228/20211228imscroll/"
    )
    fret_list = []
    n_mol = 0
    for i in range(12):
        filestr = f"L1_01_{i:02d}"
        file_name = filestr + "_traces.npz"
        category_name = filestr + "_category.npy"
        path = data_dir / file_name
        try:
            traces = ts.TimeTraces.from_npz(data_dir / file_name)
            analyzable = np.load(data_dir / category_name)
        except FileNotFoundError:
            continue
        for molecule, yes in enumerate(analyzable):
            if yes:
                donor_intensity = traces.get_intensity(
                    imp.Channel("red", "red"), molecule
                )
                acceptor_intensity = traces.get_intensity(
                    imp.Channel("red", "ir"), molecule
                )
                fret = (
                    acceptor_intensity
                    / gamma
                    / (donor_intensity + acceptor_intensity / gamma)
                )
                # fret_list.append(fret[:10].mean())
                fret_list.extend(fret[:10])
                n_mol += 1
    fret_flat = np.array(fret_list).flatten()
    fig, ax = plt.subplots()
    ax.hist(fret_flat, range=(-0.5, 1.5), bins=100, density=True)
    ax.set_xlim((0, 1))
    ax.set_xlabel("FRET")
    ax.set_ylabel("Probability density")
    ax.text(0.05, 0.9, f"N = {n_mol}", transform=ax.transAxes)
    fig.savefig(path.with_suffix(".svg"), format="svg")


if __name__ == "__main__":
    main()
