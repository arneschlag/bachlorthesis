"""some persons contained no 3d keypoint information, remove those from the dataset ...
"""

import json
import numpy as np
# '/home/scl9hi/mfsd/02_stud/01_schlag/01_datasets/07_MOTS/mot_synth_train_fixed.json'
# '/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_train_3d.json'
# '/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_validaa_3d.json'

with open('/home/scl9hi/mfsd/02_stud/01_schlag/01_datasets/07_MOTS/mot_synth_test_fixed.json.small', 'r') as data_file:
    json_data = data_file.read()

data = json.loads(json_data)

for anno in data['annotations']:
    # remove every pedestrian, that has no 3d keypoints


    datalen = len(anno['keypoints'])
    reshaped = np.array(anno['keypoints']).reshape(int(datalen/3), 3)
    num_visible = sum([kp[2] == 2 for kp in reshaped])
    visible_score = num_visible/len(reshaped)
    anno['vis_ratio'] = visible_score


with open('/home/scl9hi/mfsd/02_stud/01_schlag/01_datasets/07_MOTS/mot_synth_test_v2.json.small', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True)
