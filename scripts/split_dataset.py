#!/usr/bin/env python3

import argparse, os, sys
import csv
import tqdm

import numpy as np

def load_file(input_filename: str) -> list:

    assert(os.path.exists(input_filename))
    assert(os.path.isfile(input_filename))

    data = list()
    with open(input_filename, 'r') as ifile:
        reader = csv.DictReader(ifile)
        for r in reader:
            data.append(r)

    return data

def split_data(data:list, test_class_n=None, test_class_ratio=None) -> tuple:

    assert(len(data) > 0)
    try:
        assert((test_class_n is not None) or (test_class_ratio is not None))
    except AssertionError:
        print('Error, please define test_class_n or test_class_ratio')
        exit(1)
    except:
        raise
    if test_class_n is not None:
        assert(test_class_n > 0 and test_class_n < len(data))
    if test_class_ratio is not None:
        assert(test_class_ratio > 0.0 and test_class_ratio < 1.0)

    for row in data:
        assert('Class' in row)
        assert('DataType' in row)

    rw_data = [(i, d) for i,d in enumerate(data) if d['DataType'] == 'RW']

    classes = set([d['Class'] for d in data if d['Class'] != ''])

    class_counts = {c:0 for c in classes}
    rw_class_counts = {c:0 for c in classes}

    for d in data:
        if d['Class'] in classes:
            class_counts[d['Class']] += 1
    
    for d in rw_data:
        if d[1]['Class'] in classes:
            rw_class_counts[d[1]['Class']] += 1
    
    print(f'Class Count:')
    [print(f' - {c}: {count}') for c,count in class_counts.items()]
    print(f'\nRW Class Count:')
    [print(f' - {c}: {count}') for c,count in rw_class_counts.items()]

    test_idx_set = list()
    for c in classes:
        full_class_idx = [d[0] for d in rw_data if d[1]['Class'] == c]
        class_n = test_class_n
        if test_class_ratio is not None:
            class_n = int(np.round(test_class_ratio * len(full_class_idx)))
        test_samples = np.random.choice(full_class_idx, size=class_n, replace=False)
        print(f'Class Test Samples: {c}: {len(test_samples)}')
        for sample in test_samples:
            test_idx_set.append(sample)
    test_idx_set = set(test_idx_set)
    full_idx_set = set([d for d in range(len(data))])
    train_idx_set = full_idx_set - test_idx_set

    assert(len(train_idx_set) > 0)
    assert(len(test_idx_set) > 0)
    assert(len(test_idx_set) + len(train_idx_set) == len(full_idx_set))

    train_set = list()
    test_set = list()
    for i,d in enumerate(data):
        if i in train_idx_set:
            train_set.append(d)
        elif i in test_idx_set:
            test_set.append(d)
        else:
            raise ValueError('Split failed: Something is wrong')
    
    assert(len(train_set) > 0)
    assert(len(test_set) > 0)
    assert(len(train_set) + len(test_set) == len(data))
    return train_set, test_set
        

def output_data(train_set:list, test_set:list, output_directory: str) -> None:

    if os.path.exists(output_directory):
        assert(os.path.isdir(output_directory))
    else:
        os.path.makedirs(output_directory)

    train_filepath = os.path.join(output_directory, 'train_set.csv')
    test_filepath = os.path.join(output_directory, 'test_set.csv')

    print(f'Writing train set to: {os.path.abspath(train_filepath)}')
    with open(train_filepath, 'w') as ofile:
        writer = csv.DictWriter(ofile, fieldnames=train_set[0].keys())
        writer.writeheader()
        for d in train_set:
            writer.writerow(d)
    
    print(f'Writing test set to: {os.path.abspath(test_filepath)}')
    with open(test_filepath, 'w') as ofile:
        writer = csv.DictWriter(ofile, fieldnames=test_set[0].keys())
        writer.writeheader()
        for d in test_set:
            writer.writerow(d)



def split_label_file(input_filename:str, output_directory:str, test_class_n:int, test_class_ratio:float) -> None:

    # Load data file
    data = load_file(input_filename)

    # Split data into train and test sets
    train_set, test_set = split_data(data, test_class_n, test_class_ratio)

    # Save train and test sets to file
    output_data(train_set, test_set, output_directory)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type=str, help='CSV label file with datapaths')
    parser.add_argument('--output-directory', '-o', type=str, default='.', help='Output directory for train/test split files')
    parser.add_argument('--test-class-n', '-n', type=int, default=None, help='Choice: Define N-samples per class (same for all classes)')
    parser.add_argument('--test-class-ratio', '-r', type=float, default=None, help='Choice: Define ratio of class samples to select (different per class)')
    args = parser.parse_args()

    split_label_file(args.input_filename, args.output_directory, args.test_class_n, args.test_class_ratio)


if __name__ == '__main__':
    main()
    