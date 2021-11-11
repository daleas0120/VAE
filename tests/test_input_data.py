#!/usr/bin/env python

import unittest
import os
import sys

def project_directory():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_csv_file():
    return os.path.join(project_directory(), 'VAE_exampleDataFrame.csv')

def get_txt_file():
    return os.path.join(project_directory(), 'VAE_exampleInputFile.txt')

class TestInputData(unittest.TestCase):
        

    def test_csv_exists(self):
        self.assertTrue(get_csv_file())

    def test_txt_exists(self):
        self.assertTrue(get_txt_file())

    def test_rgb_dataset_import(self):
        from VAE.utils.VAE_utils import RGB_Dataset
        self.assertIsNotNone(RGB_Dataset)

    def test_rgb_dataset_init(self):
        # TODO Can go in its own test class for RGB dataset
        from VAE.utils.VAE_utils import RGB_Dataset
        dataset = RGB_Dataset()
        self.assertIsNotNone(dataset)

    def test_load_csv(self):
        img_dim = 256
        img_ch = 3

        from VAE.utils.VAE_utils import RGB_Dataset
        dataset = RGB_Dataset()
        orig_imgs, labels, orig_list = dataset.load_rgb(
            img_dim,
            img_ch,
            groundTruthFile=get_csv_file(),
            output_preprocess_path=None,
        )

        self.assertIsNotNone(orig_imgs)
        self.assertIsNotNone(labels)
        self.assertIsNotNone(orig_list)
        self.assertEqual(len(orig_imgs), len(labels))
        self.assertEqual(len(orig_imgs), len(orig_list))

    
    def test_load_txt(self):
        img_dim = 256
        img_ch = 3

        from VAE.utils.VAE_utils import RGB_Dataset
        dataset = RGB_Dataset()
        orig_imgs, labels, orig_list = dataset.load_rgb(
            img_dim,
            img_ch,
            groundTruthFile=get_txt_file(),
            output_preprocess_path=None,
        )

        self.assertIsNotNone(orig_imgs)
        self.assertIsNotNone(labels)
        self.assertIsNotNone(orig_list)
        self.assertEqual(len(orig_imgs), len(labels))
        self.assertEqual(len(orig_imgs), len(orig_list))

