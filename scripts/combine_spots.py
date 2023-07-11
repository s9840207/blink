from pathlib import Path
import blink.intensity_traces
import blink.image_processing
import blink.mapping



def multiple_files(datadir, mapper):
    # combine all of aoi
    for q in  datadir.iterdir():
        if q.stem[-1] != 'i':
            continue   
        aoi_data = datadir / f'{q}'
        for p in aoi_data.iterdir():
            if p.stem[0] != 'g':
                continue 
            aois = blink.image_processing.Aois.from_npz(p)
            combined = blink.intensity_traces.combine_spots(mapper, aois)
            if p.stem[0] == 'b':
                continue
            elif p.stem[0] == 'g':
                combined.to_npz(p.with_stem(p.stem[0] + '_combined_aoi'))
            else:
                combined.to_npz(p.with_stem(p.stem[0:2] + '_combined_aoi'))


def one_file(datadir, mapper):
    # combine only one aoi       
    for p in datadir.iterdir():
        if p.stem[0] != 'g':
            continue 
        aois = blink.image_processing.Aois.from_npz(p)
        combined = blink.intensity_traces.combine_spots(mapper, aois)
        combined.to_npz(p.with_stem(p.stem[0] + '_combined_aoi'))













# input the file of bgr after spot combined (combine the view at left and right side)
def main():
    datadir = Path(r'D:\CWH\2023\20230630\1_10min FRET_aoi') # the file that has been circled the spot
    mapping_path = Path(r'D:\mapping_for_cosmos_20230425\map.npz') # the mapping file that has been mapping
    mapper = blink.mapping.Mapper.from_npz(mapping_path)

    one_file(datadir, mapper)





    

if __name__ == '__main__':
    main()