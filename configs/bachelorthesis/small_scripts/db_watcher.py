import sqlite3
from nuimages.utils.utils import annotation_name, mask_decode, get_font, name_to_index_mapping
import json, tqdm

from os import listdir
from os.path import isfile, join
mypath = '/home/scl9hi/data/nuImages/original_data/nuimages/samples/CAM_FRONT'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("check images loaded")
# Replace 'path/to/your/database.db' with the actual path to your database file
db_path = "/home/scl9hi/data/nuImages/processed/nuimage_val.db"

# Connect to the database
connection = sqlite3.connect(db_path)

print("load structure ")
# Get a list of all tables in the database
cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Loop through tables and print column names for each table
for table in tables:
    table_name = table[0]
    cursor = connection.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    print(f"Table: {table_name}, Columns: {column_names}")

# create image table
images = []

print("structure loaded")
print("load images")
# Query the first entries from the '__attribute' table
cursor = connection.execute("SELECT * FROM __$frames;")
entries = cursor.fetchall()
print("images loaded")

# Print the retrieved entries
# Print the retrieved entries as dictionaries
column_names = [description[0] for description in cursor.description]


# Convert the rows to dictionaries using column names as keys
dict_entries = [dict(zip(column_names, row)) for row in entries]

good_frame_ids = []
print("filter images ...")
# Print the retrieved entries as dictionaries
for entry in tqdm.tqdm(dict_entries):
    if entry['filename'].split('/')[1] == "CAM_FRONT" and entry['frame_id'] not in good_frame_ids:
        if entry['filename'].split('/')[2] in onlyfiles:
            image = dict(file_name="./"+entry['filename'].split('/')[2],
                        height=entry['height'],
                        width=entry['width'],
                        id=entry['frame_id'],
                        license=1)
            images.append(image)
            good_frame_ids.append(entry['frame_id'])

# do it for annotations
annoatations = []
print("images filtered")


objects_ = {}
categories = []
print("get cats")
# Query the first entries from the '__attribute' table
cursor = connection.execute("SELECT * FROM __$objects;")
entries = cursor.fetchall()
print("cats")

# Print the retrieved entries
# Print the retrieved entries as dictionaries
column_names = [description[0] for description in cursor.description]


# Convert the rows to dictionaries using column names as keys
dict_entries = [dict(zip(column_names, row)) for row in entries]

print("filter cats")
for entry in tqdm.tqdm(dict_entries):
    if entry['object_id'] not in objects_:
        objects_[entry['object_id']] = entry
    if entry['object_category_name'] not in categories:
        categories.append(entry['object_category_name'])

print("get bboxs")
# Query the first entries from the '__attribute' table
cursor = connection.execute("SELECT * FROM __$boxes;")
entries = cursor.fetchall()
print("get bboxs done")

# Print the retrieved entries
# Print the retrieved entries as dictionaries
column_names = [description[0] for description in cursor.description]


# Convert the rows to dictionaries using column names as keys
dict_entries = [dict(zip(column_names, row)) for row in entries]

print("filter bboxs")
# Print the retrieved entries as dictionaries
i = 0
for entry in tqdm.tqdm(dict_entries):
    if objects_[entry['object_id']]['object_category_name'] and objects_[entry['object_id']]['object_category_name'].startswith('human') and entry["frame_id"] in good_frame_ids:
        x1, y1, x2, y2 = entry["bbox$x1"], entry["bbox$y1"], entry["bbox$x2"], entry["bbox$y2"]
        annotation = dict(area=(x2-x1)*(y2-y1),
                        category_id=1,
                        iscrowd=False,
                        image_id=entry["frame_id"],
                        bbox=[x1,y1,x2-x1,y2-y1],
                        bbox_3d=[],
                        vis_ratio=None,
                        tuncation=None,
                        keypoints=[],
                        id=i
                        )
        i +=1
        # for futher instance mask creation ...
        # rleObjs = {'size': [900, 1600], 'counts': entry['object_mask_counts']}
        # mask = mask_decode(rleObjs)
        annoatations.append(annotation)

print("filter bboxs done")
categories = [{"supercategory": "object", "id": 1, "name":  "person"}]
info = {'description': 'NuImages Dataset',
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



print("fine work done")
output = {"info": info, "licenses": licenses, "images": images, "annotations": annoatations, "categories": categories}

with open("nuimages_val.json", "w") as f:
        json_str = json.dumps(output)
        f.write(json_str)
# Close the database connection
connection.close()
