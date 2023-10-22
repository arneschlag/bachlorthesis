import os
import datetime
import json
import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(annotation_id, image_id, category_info, bounding_box, img_size):
    is_crowd = category_info['is_crowd']

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "bbox": bounding_box,
        "width": img_size[0],
        "height": img_size[1],
    }

    return annotation_info
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
]

def parse_cityperson_mat(file_path, phase):
    '''
    Citypersons .mat annotation operations
    return: a dict, key: imagename, value: an array of n* <lbl, x1 y1 x2 y2>
    '''
    tv = phase
    k = 'anno_{}_aligned'.format(phase)
    bbox_counter = 0
    rawmat = loadmat(file_path, mat_dtype=True)
    mat = rawmat[k][0]  # uint8 overflow fix
    name_bbs_dict = {}
    for img_idx in range(len(mat)):
        # each image
        img_anno = mat[img_idx][0, 0]

        city_name = img_anno[0][0]
        img_name_with_ext = img_anno[1][0]
        bbs = img_anno[2]  # n x 10 matrix
        # 10-D: n* [class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis]

        name_bbs_dict[img_name_with_ext] = bbs
        bbox_counter += bbs.shape[0]

    img_num = len(mat)  # - noins_img_counter
    print('Parsed {}: {} bboxes in {} images remained ({:.2f} boxes/img) '.format(tv,
                                                                                 bbox_counter,
                                                                                 img_num,
                                                                                 bbox_counter / img_num))
    return name_bbs_dict

def convert(phase, data_path):
    assert phase in ['train', 'val']
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    bbs_dict = parse_cityperson_mat('%s/anno_%s.mat' % (data_path, phase), phase)
    data_path2 = '/mnt/data/CityScapes/'
    fn_lst = glob.glob('%s/leftImg8bit/%s/*/*.png' % (data_path2, phase))
    positive_box_num = 0
    ignore_box_num = 0
    for image_filename in fn_lst:
        base_name = os.path.basename(image_filename)
	
        if base_name in bbs_dict:
            image = Image.open(image_filename)
            image_info = create_image_info(
                image_id, image_filename[image_filename.index('leftImg8bit'):], image.size)
            coco_output["images"].append(image_info)

            boxes = bbs_dict[base_name]
            # go through each associated annotation
            slt_msk = np.logical_and(boxes[:, 0] == 1, boxes[:, 4] >= 50)
            boxes_gt = boxes[slt_msk, 1:5]
            positive_box_num += boxes_gt.shape[0]
            for annotation in boxes_gt:
                annotation = annotation.tolist()
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = create_annotation_info(
                    annotation_id, image_id, category_info, annotation,
                    image.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1

            slt_msk = np.logical_or(boxes[:, 0] != 1, boxes[:, 4] < 50)
            boxes_ig = boxes[slt_msk, 1:5]
            ignore_box_num += boxes_ig.shape[0]
            for annotation in boxes_ig:
                annotation = annotation.tolist()
                category_info = {'id': 1, 'is_crowd': True}
                annotation_info = create_annotation_info(
                    annotation_id, image_id, category_info, annotation, image.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1

        image_id = image_id + 1
    print('positive_box_num: ', positive_box_num)
    print('ignore_box_num: ', ignore_box_num)
    with open(data_path + phase + '.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == '__main__':
    data_path = '/mnt/data/CityPersons/annotations/'
    convert(phase = 'train', data_path=data_path)
    convert(phase='val', data_path=data_path)