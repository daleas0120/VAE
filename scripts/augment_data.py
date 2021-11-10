#!/usr/bin/env python

###  I m p o r t s  ###

import argparse, os, sys
import csv
import cv2
import tqdm
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt

###  G l o b a l s  ###

AUGMENTATION_TYPES = {
    'smoothing',
    'salt-pepper',
    'gaussian',
    'rotations',
    'all',
}


###  B a s i c   F u n c t i o n s  ###


def check_augmentations(augmentation_types:list):
    for a in augmentation_types:
        if a.lower() not in AUGMENTATION_TYPES:
            raise ValueError(f'Error, unknown augmentation type: {a}')

def load_data(input_csv_filename: str) -> dict:
    
    assert(os.path.exists(input_csv_filename))
    assert(os.path.isfile(input_csv_filename))
    
    data = {
        'header': None,
        'data': list(),
    }
    with open(input_csv_filename, 'r') as ifile:
        reader = csv.DictReader(ifile)
        data['header'] = reader.fieldnames
        for r in reader:
            if np.any([not v for v in r.values()]):
                pass
            else:
                data['data'].append(r)

    return data


def create_output_directory(output_directory: str) -> None:
    if os.path.exists(output_directory):
        if os.path.isfile(output_directory):
            raise IOError(f'Error, directory is an existing file: {output_directory}')
    else:
        os.makedirs(output_directory)


def create_subdirectory(master_output_directory: str, augmentation_type: str) -> str:
    sub_dir = os.path.abspath(os.path.join(master_output_directory, augmentation_type))
    if os.path.exists(sub_dir):
        assert(os.path.isdir(sub_dir))
    else:
        os.makedirs(sub_dir)
    return sub_dir

def get_smoothing_dir(master_output_dir: str):
    return create_subdirectory(master_output_dir, 'smoothing')

def get_salt_pepper_dir(master_output_dir: str):
    return create_subdirectory(master_output_dir, 'salt-pepper')

def get_gaussian_dir(master_output_dir: str):
    return create_subdirectory(master_output_dir, 'gaussian')

def get_rotations_dir(master_output_dir: str):
    return create_subdirectory(master_output_dir, 'rotations')

def load_image(image_path: str):
    assert(os.path.exists(image_path))
    assert(os.path.isfile(image_path))
    return cv2.imread(image_path)


###  A u g m e n t a t i o n s  ###


def execute_smoothing_augmentation(data: list, master_output_directory:str, k:int=3, sigma:float=1.0) -> dict:
    """
    Applies Gaussian smoothing to input files
    """

    smoothing_dir = get_smoothing_dir(master_output_directory)

    data_cp = deepcopy(data)
    for i,d in tqdm.tqdm(enumerate(data['data']), desc='processing smoothing'):
        image_path = os.path.join(d['filePath'], d['RGB_IMG_NAME'])
        img = load_image(image_path)
        dst = cv2.GaussianBlur(img, ksize=(k,k), sigmaX=sigma)

        ext = os.path.splitext(d['RGB_IMG_NAME'])
        img_name = ext[0] + f'.k_{k}.sigma{sigma}'
        if len(ext) > 1:
            img_name += ext[1]

        output_path = os.path.join(smoothing_dir, img_name)
        data_cp['data'][i]['RGB_IMG_NAME'] = output_path
        data_cp['data'][i]['filePath'] = smoothing_dir
        cv2.imwrite(output_path, dst)


        ## Debug
        #fig, (ax1, ax2) = plt.subplots(2, 1)
        #ax1.imshow(img)
        #ax1.set_title('Normal')
        #ax2.imshow(dst)
        #ax2.set_title('Gaussian')
        #plt.show()
        #sys.exit()
    return data_cp


def execute_salt_pepper_augmentation(data: list, master_output_directory:str, ratio:float=0.25) -> dict:
    """
    Applies salt-pepper noise to input files
    """
    
    saltpepper_dir = get_salt_pepper_dir(master_output_directory)
    
    data_cp = deepcopy(data)
    for i,d in tqdm.tqdm(enumerate(data['data']), desc='processing salt-pepper'):
        image_path = os.path.join(d['filePath'], d['RGB_IMG_NAME'])
        img = load_image(image_path)
        dst = img.copy()

        r = np.random.random(size=img.shape)
        dst[r < ratio/2.0] = 0
        dst[np.logical_and(r > ratio/2.0, r < ratio)] = 255

        ext = os.path.splitext(d['RGB_IMG_NAME'])
        img_name = ext[0] + f'.sp_{ratio}'
        if len(ext) > 1:
            img_name += ext[1]
        
        output_path = os.path.join(saltpepper_dir, img_name)
        data_cp['data'][i]['RGB_IMG_NAME'] = output_path
        data_cp['data'][i]['filePath'] = saltpepper_dir
        cv2.imwrite(output_path, dst)

        ## Debug
        #fig, (ax1, ax2) = plt.subplots(2, 1)
        #ax1.imshow(img)
        #ax1.set_title('Normal')
        #ax2.imshow(dst)
        #ax2.set_title('Salt-Pepper')
        #plt.show()
        #sys.exit()
    return data_cp


def execute_gaussian_augmentation(data: list, master_output_directory:str) -> dict:
    """
    Applies Gaussian nosie to input files
    """
    
    gaussian_dir = get_gaussian_dir(master_output_directory)

    data_cp = deepcopy(data)
    for i,d in tqdm.tqdm(enumerate(data['data']), desc='processing gaussian noise'):
        image_path = os.path.join(d['filePath'], d['RGB_IMG_NAME'])
        img = load_image(image_path)
        dst = img.copy().astype('int32')

        r = np.random.normal(loc=0.0, scale=np.std(img), size=dst.shape)
        #r = np.random.normal(loc=0.0, scale=1.0, size=dst.shape)
        dst += np.round(r).astype('int32')
        dst[dst < 0] = 0
        dst[dst > 255] = 255
        dst = dst.astype('uint8')

        ext = os.path.splitext(d['RGB_IMG_NAME'])
        img_name = ext[0] + f'.gaussian'
        if len(ext) > 1:
            img_name += ext[1]
        
        output_path = os.path.join(gaussian_dir, img_name)
        data_cp['data'][i]['RGB_IMG_NAME'] = output_path
        data_cp['data'][i]['filePath'] = gaussian_dir
        cv2.imwrite(output_path, dst)

        ## Debug
        #fig, (ax1, ax2) = plt.subplots(2, 1)
        #ax1.imshow(img)
        #ax1.set_title('Normal')
        #ax2.imshow(dst)
        #ax2.set_title('Gaussian Noise')
        #plt.show()
        #sys.exit()
    return data_cp


def execute_rotations_augmentation(data: list, master_output_directory:str, rotation_degrees:float=22.5) -> dict:
    """
    Applies rotations to input files
    """
    rotations_dir = get_rotations_dir(master_output_directory)
    
    data_cp = {
        'header': data['header'],
        'data': list()
    }
    for r_deg in tqdm.tqdm(np.arange(rotation_degrees, 360.0, rotation_degrees), desc='processing rotations'):
        for i, d in enumerate(data['data']):
            image_path = os.path.join(d['filePath'], d['RGB_IMG_NAME'])
            img = load_image(image_path)
            (h,w) = img.shape[0:2]
            center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(center, r_deg, 1.0)
            dst = cv2.warpAffine(img, M, (w, h))

            ext = os.path.splitext(d['RGB_IMG_NAME'])
            img_name = ext[0] + f'.r_deg_{r_deg}'
            if len(ext) > 1:
                img_name += ext[1]
            
            output_path = os.path.join(rotations_dir, img_name)
            dc = deepcopy(data['data'][i])
            dc['RGB_IMG_NAME'] = output_path
            dc['filePath'] = rotations_dir
            data_cp['data'].append(dc)
            cv2.imwrite(output_path, dst)

            ## Debug
            #fig, (ax1, ax2) = plt.subplots(2, 1)
            #ax1.imshow(img)
            #ax1.set_title('Normal')
            #ax2.imshow(dst)
            #ax2.set_title('rotated')
            #plt.show()
            #sys.exit()
    return data_cp


def write_master_label(data_list: list, output_directory:str) -> None:
    
    assert(len(data_list) > 0)

    augmented_filename = os.path.join(output_directory, 'augmented_labels.csv')
    output_filename = os.path.join(output_directory, 'full_labels.csv')

    fieldnames = data_list[0]['header']

    with open(augmented_filename, 'w') as aofile, open(output_filename, 'w') as ofile:
        o_writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        o_writer.writeheader()
        a_writer = csv.DictWriter(aofile, fieldnames=fieldnames)
        a_writer.writeheader()
        for i,d in tqdm.tqdm(enumerate(data_list), 'Writing label files'):
            o_writer.writerows(d['data'])
            if i != 0:
                a_writer.writerows(d['data'])


def execute_augmentations(data: dict, augmentation_types: list, output_directory: str) -> None:
    data_list = [data]
    for a in tqdm.tqdm(augmentation_types, desc='Processing augmentations'):
        augmentation_type = a.lower()
        if augmentation_type == 'smoothing' or augmentation_type == 'all':
            data_list.append(execute_smoothing_augmentation(data, output_directory))
        if augmentation_type == 'salt-pepper' or augmentation_type == 'all':
            data_list.append(execute_salt_pepper_augmentation(data, output_directory))
        if augmentation_type == 'gaussian' or augmentation_type == 'all':
            data_list.append(execute_gaussian_augmentation(data, output_directory))
        if augmentation_type == 'rotations' or augmentation_type == 'all':
            data_list.append(execute_rotations_augmentation(data, output_directory))
    write_master_label(data_list, output_directory)


###  E n t r y   F u n c t i o n s  ###


def do_augmentations(input_csv_filename: str, augmentation_types: list, output_directory: str) -> None:

    print('Check Augmentation Types')
    check_augmentations(augmentation_types)

    print('Load CSV data')
    data = load_data(input_csv_filename)

    print(f'Create output directory: {output_directory}')
    create_output_directory(output_directory)

    print(f'Executing augmentations: {augmentation_types}')
    execute_augmentations(data, augmentation_types, output_directory)


def main():

    default_output_directory = os.path.abspath(os.path.join('.', 'augmented_data'))
    default_augmentation = ['all']

    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv_filename', type=str, help='Input labels files and data paths')
    parser.add_argument('--output-directory', '-o', type=str, default=default_output_directory, help=f'Output directory (default={default_output_directory})')
    parser.add_argument('--augmentation-types', '-a', type=str, nargs='+', default=default_augmentation, choices=AUGMENTATION_TYPES, help=f'Augmentation type (default={default_augmentation})')
    args = parser.parse_args()

    do_augmentations(args.input_csv_filename, args.augmentation_types, args.output_directory)


if __name__ == '__main__':
    main()
    