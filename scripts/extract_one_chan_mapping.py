import numpy as np
from pathlib import Path



path = Path(r'D:\mapping') # mapping_file的位置
npz_path = path / 'map.npz'# mapping_file的名子
new_npz_name = ['blue-green', 'blue-red', 'green-red']
with np.load(npz_path) as npz_file:
    for name in new_npz_name:
        items = {}
        items[name] = npz_file[name] 
        np.savez(path / f'map{name}', **items)
# with np.load(npz_path) as npz_file:
#     print(npz_file.files)
#     breakpoint()