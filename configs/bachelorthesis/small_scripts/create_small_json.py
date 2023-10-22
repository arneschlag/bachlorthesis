"""this file will create a small dataset given the main datasets.
"""

import json
import sys, getopt

from joblib import Parallel, delayed
import random


#opts, args = getopt.getopt(sys.argv[1:],"i:",["ifile="])
paths = ["/home/scl9hi/mfsd/02_stud/01_schlag/01_datasets/02_mscoco/person_keypoints_train2017_fixed.json",
         "/home/scl9hi/mfsd/02_stud/01_schlag/01_datasets/02_mscoco/person_keypoints_val2017_fixed.json"]

#path = opts[0][1]
#opts[0][1] = "/mnt/data/GTA/JTA-Dataset/coco_out/train/jta_train_fixed.json"

def create_smaller(path):
    print("loading",path)
    with open(path, 'r') as data_file:
        json_data = data_file.read()

    data = json.loads(json_data)
    print("processing")
    random.seed(10)
    if len(data['images']) > 1000:
        imgs = random.sample(data['images'], 1000)
    else:
        imgs = data['images']
    if len(data['images']) <= 1000:
        print(path, " is to small")
    #imgs = data['images'][:1000]
    ids = list(img['id'] for img in imgs)

    annos = []
    for anno in data['annotations']:
        if anno['image_id'] in ids:
            annos.append(anno)
    data['images'] = imgs
    data['annotations'] = annos

    print("saveing")
    with open(path+".small", 'w', encoding ='utf8') as json_file:
        json.dump(data, json_file, allow_nan=True)

# results = Parallel(n_jobs=32)(
#     delayed(create_smaller)(
#         path
#     )
#     for path in paths
# )

for path in paths:
    create_smaller(path)
