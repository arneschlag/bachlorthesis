import json
import sys, getopt
import numpy as np
from joblib import Parallel, delayed
import random
from bolt.applications.scl9hi.preprocessing.datasets import Dataset, Datasets
import pickle
import matplotlib.pyplot as plt

def calculate_stats_split(data):
    num_of_imgs = len(data["images"])
    num_keypoints = 0
    if len(data["annotations"]) > 0 and "keypoints" in data["annotations"][0]:
        for kp in data["annotations"]:
            if len(kp["keypoints"]) > 0:
                num_keypoints = int(len(kp["keypoints"])/3)
    count_dic = {}
    total_area = []
    total_width = []
    total_height = []
    vis_ratio = []
    truncation = []
    keypoints_3d_x = []
    keypoints_3d_y = []
    keypoints_3d_z = []
    kp_visible = np.zeros(num_keypoints)
    valid_person_counter =0

    tags = ["pedestrian", "person", "pedestrain"]

    ids_to_search_for = []
    for entry in data["categories"]:
        if entry["name"] in tags:
            ids_to_search_for.append(entry['id'])



    for entry in data["annotations"]:

        if entry["category_id"] not in count_dic.keys():
            count_dic[entry["category_id"]] = 1
        else:
            count_dic[entry["category_id"]] += 1
        if entry["category_id"] in ids_to_search_for:
            if 'area' not in entry.keys():
                area = entry['bbox'][2] * entry['bbox'][3]
            else:
                area = entry['area']
            total_area.append(area)
            total_width.append(entry['bbox'][2])
            total_height.append(entry['bbox'][3])

            if "vis_ratio" in entry.keys():
                vis_ratio.append(entry["vis_ratio"])

            if "truncation" in entry.keys():
                truncation.append(entry["truncation"])

            if "keypoints" in entry.keys() and len(entry["keypoints"]) != 0:
                if len(entry["keypoints"])%3 != 0:
                    raise KeyError
                valid_person_counter+=1

                # get visibilty of keypoints
                if len(entry['keypoints'])%3 == 0:
                    # has visibilty flag
                    reshaped = np.array(entry['keypoints']).reshape(num_keypoints,3)
                    visible_kps = np.array(np.rot90(reshaped)[0] == 2, dtype=bool)
                    kp_visible = kp_visible + visible_kps
            if "keypoints_3d" in entry.keys() and len(entry["keypoints_3d"]) != 0:
                reshaped = np.array(entry['keypoints_3d'])
                if len(reshaped.shape) == 1:
                    if len(entry['keypoints_3d']) == 88:
                        reshaped = reshaped.reshape(num_keypoints, 4)
                        a, z, y, x = np.mean(np.rot90(reshaped), axis=1)
                    else:
                        reshaped = reshaped.reshape(num_keypoints, 3)
                        z, y, x = np.mean(np.rot90(reshaped), axis=1)
                else:
                    z, y, x = np.mean(np.rot90(reshaped), axis=1)
                keypoints_3d_x.append(x)
                keypoints_3d_y.append(y)
                keypoints_3d_z.append(z)


    if valid_person_counter > 0:
        kp_visible = [kp/valid_person_counter for kp in kp_visible]
        avg_kp_visible = sum(kp_visible)/num_keypoints
    else:
        kp_visible = np.zeros(num_keypoints)
        avg_kp_visible = 0


    num_pedestrians = sum([number if id_ in ids_to_search_for else 0 for id_, number in count_dic.items()])

    return dict(total_num=dict(num_images=num_of_imgs,
                    num_keypoints=num_keypoints,
                    num_pedestrians=num_pedestrians
                ),
                all_values=dict(
                    total_area=total_area,
                    total_width=total_width,
                    total_height=total_height,
                    vis_ratio=vis_ratio,
                    truncation=truncation,
                    keypoints_3d_x=keypoints_3d_x,
                    keypoints_3d_y=keypoints_3d_y,
                    keypoints_3d_z=keypoints_3d_z
                ),
                avg=dict(
                    kp_visible=kp_visible,
                    avg_kp_visible=avg_kp_visible
                )
                )



def extract_information(name: str, dataset: Dataset):
    if not name.startswith('__'):
        dataset_info = {}
        # if dataset.has_debug:
        #     add =".small"
        # else:
        #     add = ""
        add = ""
        if dataset.splits.train:
            with open(dataset.splits.train.annotation_path+add, 'r') as data_file:
                train_data = data_file.read()
                dataset_info["train"] = calculate_stats_split(json.loads(train_data))
        if dataset.splits.val:
            with open(dataset.splits.val.annotation_path+add, 'r') as data_file:
                val_data = data_file.read()
                dataset_info["val"] = calculate_stats_split(json.loads(val_data))
        if dataset.splits.test:
            with open(dataset.splits.test.annotation_path+add, 'r') as data_file:
                test_data = data_file.read()
                dataset_info["test"] = calculate_stats_split(json.loads(test_data))
        print("done:", name)
        return name, dataset_info
print("starte")
all_datasets = list(vars(Datasets).items())[2:14]
#extract_information(dataset_name, dataset)

# results0 = Parallel(n_jobs=len(all_datasets))(
# #results = Parallel(n_jobs=1)(
#     delayed(extract_information)(
#         dataset_name, dataset
#     )
#     for dataset_name, dataset in all_datasets
# )
# print("speichere")
# with open('results.pickle', 'wb') as handle:
#     pickle.dump(results0, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('results.pickle', 'rb') as handle:
    results = pickle.load(handle)
print("loaded")



import matplotlib.pyplot as plt
fig = plt.figure(figsize =(10, 7))
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


normalize_x = [1920, 1920, 640, 3678, 1920, 2048, 1920, 1280, 1920, 1280, 1600, 3840]
normalize_y = [1280, 1280, 480, 2688, 1080, 1024, 1080, 720, 1080, 800, 900, 2160]

keys = ['total_area', 'total_width', 'total_height', 'vis_ratio', 'truncation', 'keypoints_3d_x', 'keypoints_3d_y', 'keypoints_3d_z']
normalize = [True, True, True, False , False, False, False, False]
for norm, key in zip(normalize, keys):
    fig = plt.figure(figsize =(12, 7))
    data = []
    for i, result in enumerate(results):
        if len(result) > 2:
            set_ =result[1]['train']['all_values'][key] + result[1]['val']['all_values'][key] + result[1]['test']['all_values'][key]
        else:
            set_ = result[1]['train']['all_values'][key] + result[1]['val']['all_values'][key]
        if norm:
            if key == "total_width":
                set_ = [val/normalize_x[i] for val in set_]
            if key == "total_height":
                set_ = [val/normalize_y[i] for val in set_]
            if key == "total_area":
                set_ = [val/(normalize_y[i]*normalize_x[i]) for val in set_]
        # filter out None
        set_ = [val for val in set_ if val is not None]
        data.append(set_)
    data = [np.array(set_)[~is_outlier(np.array(set_))] if len(set_) > 0 else [] for set_ in data]
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 0)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)

    # x-axis labels
    ax.set_yticklabels(['KI_A', 'KI_AS', 'MSCOCO', 'PEDX', 'JTA', 'CITYPERSONS', 'MOT17', 'BDD100K', 'MOTS', 'SHIFT', 'NUIMAGES', 'AGORA'])

    # Adding title
    plt.title(f'Boxplot of {key}')

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.show()
    plt.savefig(f'boxplot_{key}.png')
