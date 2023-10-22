from pedx.data_loader import *
from shapely import geometry
import os.path
import glob
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import bz2
import pickle
import _pickle as cPickle

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

results = decompress_pickle('pedx.pbz2')
dates = list_all_capture_dates()
cameras = list_all_camera_names()
basedir = '/mnt/data/pedx/data'
n_jobs = 64

dataset = {'info': {
        'description': 'PedX Dataset',
        'url': 'https://pedx.io',
        'version': '1.0',
        'year': 2023,
        'contributor': 'Robert Bosch GmbH',
        'date_created': '2023/06/12'},
    'licenses': [
        {
            'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
            'id': 1,
            'name': 'Edit this license'
        }],
    'images': [] ,
    'annotations': [], 
    'categories': [
        {'supercategory': 'object', 'id': 0, 'name': 'unlabeled'},
        {'supercategory': 'object', 'id': 2, 'name': 'person'}]
}

anno_id = 0
annotations = []
print("Generating Jobs")
jobs = []
for date in dates:
    for camera in cameras:
        fns = glob.glob(os.path.join(basedir, 'labels/2d', date, '*.json'))
        for frame_id in range(0, len(fns)+10):
            jobs.append([date, camera, frame_id])
print("Generating Jobs done")

kp_list =['nose', 'leye', 'reye', 'head', 'lknee', 'neck', 'lwri', 'lankl', 'lhip', 'mouth',  'rwri', 'rknee', 'lelb', 'lsho', 'rhip', 'rankl', 'relb', 'rsho']
def create_annotation_from_data(data, frame_id, anno_id):
    annotation = {}

    # calculate area
    if data['category'] == 'pedestrian':
        poly = geometry.Polygon(data['polygon'])
        annotation['area'] = poly.area
        annotation['segmentation'] = [] # TODO
        annotation['category_id'] = 2
        annotation['is_crowd'] = False

        x1, y1, x2, y2 = list(poly.bounds)
        annotation['bbox'] = [x1, y1,  abs(x2-x1), abs(y2-y1)]
        annotation['bbox_3d'] = [] # TODO
        annotation['vis_ratio'] = 1.0 # TODO calculate real vis_ratio
        annotation['truncation'] = 0 # TODO 
        annotation['keypoints_3d'] = []
        annotation['num_keypoints'] = len(kp_list)
        annotation['image_id'] = frame_id # dont use data['frame_id'] because it may be associated with more than one image
        annotation['id'] = anno_id
        annotation['person'] = data['tracking_id']
        if data['keypoint'] is None:
            annotation['keypoints'] = np.zeros(len(kp_list)*3).tolist()
        else:
            anno_list = []
            for key in kp_list:
                if key in data['keypoint']:
                    anno_list += [int(data['keypoint'][key]['x']), int(data['keypoint'][key]['y']), 2 if data['keypoint'][key]['visible'] else 1]
                else:
                    anno_list += [0,0,0]
            annotation['keypoints'] = anno_list
        return annotation
    else:
        print(data['category'])

# Filter Results, so those Image which have no Annotations get removed
results_a = [(i, j) for i, j in zip(results, jobs) if len(i.keys())>0]
anno_id = 0
frame = 0
# generate images list
images = []
annotations = []
for metadata, image_info in results_a:
    date, camera, frame_id = image_info

    nullen = "0" * (7 - len(str(frame_id)))  # Erstellt eine Anzahl von Nullen, die der Differenz zwischen 7 und der Länge der Variable entspricht
    formated_number = nullen + str(frame_id)  # Fügt die Nullen dem String hinzu

    image_name = f"{date}/{camera}/{date}_{camera}_{formated_number}.jpg"
    width, height = get_image_shape(camera)
    # image information
    image_info = {
        "license": 3,
        "file_name": image_name,
        "height": height,
        "width": width,
        "date_captured": date,
        "coco_url": "none",
        "flickr_url": "none",
        "real_frame_name": frame_id,
        "id": frame
    }

    images.append(image_info)

    #annotation
    for annotation_object in metadata:
        annotations.append(create_annotation_from_data(metadata[annotation_object], frame, anno_id))
        anno_id += 1
    frame += 1

train_dataset = dataset
train_dataset["images"] = images
train_dataset["annotations"] = annotations

compressed_pickle("all_dataset", train_dataset)