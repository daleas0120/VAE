# Variational Auto Encoder

Based on the implementation by F. Chollet found here:
- https://keras.io/examples/generative/vae/
- https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py

## Installation
~~Implemented using Python 3.7 and Tensorflow 2.3.0.  Uses GPU when available.  See the **_vae37.yml_** file (Windows 10 compatible)
for list of packages.  Although this uses a conda environment, there were issues using the
Tensorflow packages available through conda, solved by mixing pip packages and conda packages: pip for tensorflow and
 keras, conda for everything else possible.~~

This project uses Python 3.7 and Tensorflow 2.3.0, but work is in progress to update to the most latest version of Tensorflow. Environments are tracked using Anaconda, and an assortment of cross-platform environment scripts are provided under the `environments` directory. The earliest working environment is provided under the `vae37.yml` configuration file. An environment script for the latest Tensorflow version is provided under `environment_latest.yaml`.  Tests are written to ensure the environments are working as expected.

### TensorFlow 2.6 notes

Prior to Tensorflow 2.6, packages for `tensorflow`, `tensorflow-gpu`, `tensorboard`, `keras`, and build tools and drivers were installed using a mixture of conda dependencies and pip dependencies.  Post 2.6, `tensorflow-gpu` and `tensorboard` are privided via a pip install of `tensorflow`.  Installation of `cudatoolkit` and `cdnn` are still handled with conda, and `keras` must be installed separately with pip matching the `tensorflow` version. 

## Running Tests with Nosetests

We use `nosetests` to ensure that the environment is installed correctly and the code is running as expected.

To run unit tests, execute the following:

```bash
nosetests -s
```

This will execute the test scripts found in the `tests`. Adding the `-s` flag to suppress standard output. 

### Writing Tests

When adding a new class, module, and function, it is important to write tests for it under the `tests` directory. Each test script will be prepended with `test_` for the `nosetests` to find. For now, the most simple way to implement a test is to use the `unittest` package.  An example `test_example.py` is shown below:

```python
import unittest

class ExampleTest(unittest.TestCase):
    def test_a_is_not_none(self):
        a = 1
        self.assertIsNotNone(a)
    def test_a_is_one(self):
        a = 1
        self.assertIsEqual(a, 1)
    def test_import_sys(self):
        import sys
        self.assertIsNotNone(sys)
```

When writing tests, we want to make sure each test function tests exactly one thing.  There should be exactly one test assertion for each function, and not a mix of assertions.  For each method in a `unittest.TestCase` class, we prepend the method with `test_<testname>`, similar to how we name the python script itself.  Each method takes a `self` object of the `unittest.TestCase` inherited class, and assertions are called directly on `self`.  Standard `assert` also works. When we get into the mix of testing `numpy` arrays, we must import Numpy's testing package `numpy.testing` and use its methods on arrays and matrices. 

## Basic Usage
### Data:
Example data may be found in the `data/` folder (extract the included data.zip folder).  Network currently accepts pandas
data frames as .csv files, and a custom .txt file format (example also provided).  Legacy code
in the `RGB_Dataset()` class is capable of parsing .json files formatted after the MSCOCO data structure, but this has
not really been developed.  Pandas data frames are the way to go.

Images are downsampled to 128x128x3, and contain a single object instance. Images
 were pre-padded to maintain aspect ratio for best results.

### To Train:
`python train_vae.py --labels=VAE_exampleDataFrame.csv
`

Default values are found in the `main.py` argument parser.

### To Evaluate Test Data:
`python test_vae.py --encoderPath=PATH\TO\ENCODER\FOLDER --decoderPath=PATH\TO\DECODER\FOLDER
--labels=PATH\TO\CSV
`
### To use Tensorboard:
`python -m tensorboard.main --logdir=PATH\TO\LOG_DIR`