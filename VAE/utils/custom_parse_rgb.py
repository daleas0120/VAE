#!/usr/bin/env python

# Custom Text Parser

import ntpath
import os
from matplotlib import collections
import tqdm

# For input files that begin by specifying the root data directory, a PLATFORM variable
# can be defined for use in the load() function on line 49

PLATFORM = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..', '..'))

def getClassName(n):
    """
    Maps first digit of integer image label to class name
    n: int image label
    returns: Class name as string
    """
    class_idx = n%10

    if class_idx == 1:
        return 'plane'
    elif class_idx == 2:
        return 'glider'
    elif class_idx == 3:
        return 'kite'
    elif class_idx == 4:
        return 'quadcopter'
    elif class_idx == 5:
        return 'eagle'
    else:
        return 'None'

def getDataType(n):
    """
    Maps second digit of integer img label to data type (real data or synthetic data)
    n: int image label
    returns: string label indicating RW or VW label
    """

    type_idx = n//10
    if type_idx == 0:
        return 'RW'
    else:
        return 'VW'


def load_csv(filepath):
    import pandas as pd
    annotations = {}
    df = pd.read_csv(filepath, skip_blank_lines=True)
    df = df.dropna()

    for index, row in tqdm.tqdm(df.iterrows(), desc='loading csv'):
        d = {}  # creates dict for this file
        img_dir = row.filePath
        rgb_img = row.RGB_IMG_NAME
        z_img = row.Z_IMG_NAME
        img_path = os.path.join(img_dir, rgb_img)
        z_img_path = os.path.join(img_dir, z_img)
        d['filename'] = row.RGB_IMG_NAME
        d['z_filename'] = row.Z_IMG_NAME
        d['img_path'] = img_path
        d['z_img_path'] = z_img_path
        d['Class'] = row.Class
        d['DataType'] = row.DataType
        annotations.update({index:d})

    return [annotations, row.filePath]

def load(filepath: object) -> object:
    nested_dict = lambda: collections.defaultdict(nested_dict)
    annotations = {}
    # open the file and read through it line by line

    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line[0] == "#" or line == "\n":
            line = file_object.readline()
        paths = line.split(",")
        num_paths = len(paths)
        for x in paths:
            x.strip()
        line = file_object.readline()
        #platform = os.environ['PLATFORM']
        platform = PLATFORM
        print("Platform: "+platform)
        if platform == "SL02319_WIN":
            img_dir = paths[0]
        elif platform == "SL02319":
            img_dir = paths[2]
        elif platform == "SL023L1_WIN":
            img_dir = paths[2]
        elif platform == ("LAMBDA1" or "LAMBDA2"):
            img_dir = paths[2]
        else:
            img_dir = os.path.join(platform, paths[1])

        print('Getting data from '+img_dir)
        img_dir.strip("\n")

        while line:
            if len(line) > 0 and line[0] != "#" and line != "\n":
                d = {}  # creates dict for this file

                parts = line.split("{")
                img_files = parts[0].split(",")

                rgb_img = img_files[0]
                rgb_img.strip()  # remove trailing white spaces
                path, name = ntpath.split(rgb_img)
                img_path = os.path.join(img_dir, rgb_img)
                img_size = os.path.getsize(img_path)

                d['filename'] = name
                d['size'] = img_size
                d['img_path'] = img_path
                d['z_filename']= 'None'

                parts.remove(parts[0])

                dataTypeList = []
                gt_boxes = []

                while len(parts) != 0:

                    tmp_data = parts[0].split("}")
                    tmp_data = tmp_data[0].split(",")
                    label = int(tmp_data[len(tmp_data) - 1])

                    d['Class'] = getClassName(label)
                    d['DataType'] = getDataType(label)

                    tmp_data.remove(tmp_data[len(tmp_data) - 1])

                    tmp_dict = {}
                    shape_attributes = {'name': "polygon"}

                    if len(tmp_data) > 0:
                        x_coords=[]
                        y_coords=[]
                        if len(tmp_data) == 4:
                            x_1 = int(tmp_data[0])
                            y_1 = int(tmp_data[1])
                            x_2 = int(tmp_data[2]) + x_1
                            y_2 = int(tmp_data[3]) + y_1
                            x_coords = [x_1, x_2, x_1, x_2]
                            y_coords = [y_1, y_1, y_2, y_2]
                        else:
                            for i in range(int(len(tmp_data)/2)):
                                x_coords.append(int(tmp_data[0]))
                                tmp_data.remove(tmp_data[0])
                            while len(tmp_data) != 0:
                                y_coords.append(int(tmp_data[0]))
                                tmp_data.remove(tmp_data[0])
                            x_1 = min(x_coords)
                            y_1 = min(y_coords)
                            x_2 = max(x_coords)
                            y_2 = max(y_coords)

                        if len(x_coords) != len(y_coords):
                            print("ERROR: Number of x_coords does not equal number of y_coords")
                            return

                        shape_attributes['all_points_x'] = x_coords
                        shape_attributes['all_points_y'] = y_coords
                        tmp_dict['shape_attributes'] = shape_attributes

                        gt_boxes.append([y_1, x_1, y_2, x_2])

                    dataTypeList.append(label)

                    d.update({'DataType': dataTypeList})
                    parts.remove(parts[0])

                    annotations.update({img_path:d})

            line = file_object.readline()
    data = [annotations, img_dir]
    return data