# Variational Auto Encoder

Based on the implementation by F. Chollet found here:
- https://keras.io/examples/generative/vae/
- https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py

## Installation
Implemented using Python 3.7 and Tensorflow 2.3.0.  Uses GPU when available.  See the **_vae37.yml_** file (Windows 10 compatible)
for list of packages.  Although this uses a conda environment, there were issues using the
Tensorflow packages available through conda, solved by mixing pip packages and conda packages: pip for tensorflow and
 keras, conda for everything else possible.

## Basic Usage
#### Data:
Example data may be found in the `data/` folder (extract the included data.zip folder).  Network currently accepts pandas
data frames as .csv files, and a custom .txt file format (example also provided).  Legacy code
in the `RGB_Dataset()` class is capable of parsing .json files formatted after the MSCOCO data structure, but this has
not really been developed.  Pandas data frames are the way to go.

Images are downsampled to 128x128x3, and contain a single object instance. Images
 were pre-padded to maintain aspect ratio for best results.

#### To Train:
`python main.py --labels=VAE_exampleDataFrame.csv
`

Default values are found in the `main.py` argument parser.

#### To Test:
`python test.py --encoderPath=PATH\TO\ENCODER\FOLDER --decoderPath=PATH\TO\DECODER\FOLDER
--labels=PATH\TO\CSV
`
#### To use Tensorboard:
`python -m tensorboard.main --logdir=PATH\TO\LOG_DIR`