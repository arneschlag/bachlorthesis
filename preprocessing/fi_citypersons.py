path = "/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/04_CityPersons/train_citypersons.json"
import json
import numpy as np

with open(path, 'r') as f:
    data = json.load(f)
data['categories'][0]['id'] = 2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open('/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/04_CityPersons/train_citypersons_fixed.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)