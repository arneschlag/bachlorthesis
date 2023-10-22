from pycocotools.coco import COCO
import numpy as np
import cv2, os, json
basepath = "/mnt/data/MOT_Synth/"
annotation_path = f'{basepath}annotations/'
basepath_instance_mask_out = f'{basepath}instance_mask/'
from joblib import Parallel, delayed
import bz2
import pickle
import _pickle as cPickle

info = {'description': 'KI-Absicherung Dataset',
    'url': 'http://myurl.org',
    'version': '1.0',
    'year': 2023,
    'contributor': 'Robert Bosch GmbH',
    'date_created': '2023/05/13'
}
licenses = [
    {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
    'id': 1, 
    'name': 'Edit this license'}
]

motsynth_categories = []
annotations = []
loc_images = []

train = list(range(0, 613))
val = list(range(613, 691))
test = list(range(691, 768))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def process(sequence_file):
    coco_annotation = COCO(annotation_path+sequence_file)
    for cat in coco_annotation.cats:
        if coco_annotation.cats[cat] not in motsynth_categories:
            motsynth_categories.append(coco_annotation.cats[cat])

    sequence = int(sequence_file.split('.')[0])
    if sequence > 524:
        sequence=sequence-1
    if sequence > 629:
        sequence=sequence-1
    if sequence > 652:
        sequence=sequence-1
    if sequence > 757:
        sequence=sequence-1
    framenr = 1
    internal_imgs = []
    internal_anno = []
    for img in coco_annotation.imgs:
        # convert the org image name to our img name
        real_framenr = int(framenr/20)+1
        if ((framenr-1) % 20) == 0:
            ann_ids = coco_annotation.getAnnIds(imgIds=[img], iscrowd=None)
            anns = coco_annotation.loadAnns(ann_ids)


            coco_annotation.imgs[img]['file_name'] = f'./seq_{sequence}_{real_framenr:06d}.np.lz4'

            # TODO mod 20 on image nr .... and compare
            internal_imgs.append(coco_annotation.loadImgs(img)[0])
            #mask = np.zeros((internal_imgs[0]['height'],internal_imgs[0]['width'])).astype(np.uint8)
            if len(anns) >= 255:
                print("ALARM")
                return None
            for i in range(len(anns)):
                anns[i]['local_id'] = i+1
                #mask += coco_annotation.annToMask(anns[i])*(i+1)
                anns[i].pop('segmentation')
            #cv2.imwrite(f'{basepath_instance_mask_out}seq_{sequence}_{real_framenr:06d}.png', mask)
            internal_anno += anns
        framenr +=1
    return sequence, internal_imgs, internal_anno

arr = os.listdir(annotation_path)
results = Parallel(n_jobs=1)(
    delayed(process)(sequence_file)
    for sequence_file in arr
)
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
# train
print("Dump Data ...")
for sequence, imgs, anno in results:
    if sequence in train:
        annotations += anno
        loc_images += imgs
data = {"info": info, "licenses": licenses, "images": loc_images, "annotations": annotations, "categories": [dict(supercategory="none", id=1, name="pedestrian", keypoints=['Head_top', 'Head', 'Neck', 'R_Clavicle', 'R_UpperArm','R_Forearm', 'R_Hand', 'L_Clavicle', 'L_UpperArm', 'L_Forearm', 'L_Hand', 'Spine3', 'Spine2', 'Spine1', 'Spine0', 'Spine_Root', 'R_Thigh', 'R_Calf', 'R_Foot', 'L_Thigh', 'L_Calf', 'L_Foot'], skeleton=[[1, 2], [2, 3], [3, 4], [3, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [16, 20], [20, 21], [21, 22]])]}
compressed_pickle('mot_synth_train.pbz2', data) 
with open('mot_synth_train.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)


annotations = []
loc_images = []
for sequence, imgs, anno in results:
    if sequence in val:
        annotations += anno
        loc_images += imgs
# validation
data = {"info": info, "licenses": licenses, "images": loc_images, "annotations": annotations, "categories": [dict(supercategory="none", id=1, name="pedestrian", keypoints=['Head_top', 'Head', 'Neck', 'R_Clavicle', 'R_UpperArm','R_Forearm', 'R_Hand', 'L_Clavicle', 'L_UpperArm', 'L_Forearm', 'L_Hand', 'Spine3', 'Spine2', 'Spine1', 'Spine0', 'Spine_Root', 'R_Thigh', 'R_Calf', 'R_Foot', 'L_Thigh', 'L_Calf', 'L_Foot'], skeleton=[[1, 2], [2, 3], [3, 4], [3, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [16, 20], [20, 21], [21, 22]])]}
compressed_pickle('mot_synth_val.pbz2', data) 
with open('mot_synth_val.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)

annotations = []
loc_images = []
for sequence, imgs, anno in results:
    if sequence in test:
        annotations += anno
        loc_images += imgs
# test
data = {"info": info, "licenses": licenses, "images": loc_images, "annotations": annotations, "categories": [dict(supercategory="none", id=1, name="pedestrian", keypoints=['Head_top', 'Head', 'Neck', 'R_Clavicle', 'R_UpperArm','R_Forearm', 'R_Hand', 'L_Clavicle', 'L_UpperArm', 'L_Forearm', 'L_Hand', 'Spine3', 'Spine2', 'Spine1', 'Spine0', 'Spine_Root', 'R_Thigh', 'R_Calf', 'R_Foot', 'L_Thigh', 'L_Calf', 'L_Foot'], skeleton=[[1, 2], [2, 3], [3, 4], [3, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [16, 20], [20, 21], [21, 22]])]}
compressed_pickle('mot_synth_test.pbz2', data) 
with open('mot_synth_test.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)