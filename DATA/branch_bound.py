import numpy as np

class TSPReader:

    def __init__(self, path, n_meta=4):
        with open(path, 'r') as f:
            lines = f.readlines()

        assert len(lines) > n_meta and lines[n_meta].strip() == "NODE_COORD_SECTION"
        meta_info = lines[:n_meta]

        for x, y in [x.split(':') for x in meta_info]:
            setattr(self, x.lower().strip(), y.strip())

        coords = lines[n_meta+1:]
        coords = np.fromstring(coords, dtype=float)
        self.city_indices = coords[:, 0].astype(int)
        self.city_coords = coords[:, 1:]


        print(meta_info)
        print(self.city_indices)
        print(self.city_coords)



TSPReader("DATA/Atlanta.tsp")