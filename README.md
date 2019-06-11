# Handwriting Recognition course, University of Groningen
## Group 7

Authors:

- Stefan Evanghelides - s2895323

- Huaitin Lu - s3339246

- Sudhakaran Jain - s3558487

- Subilal Vattimunda Purayil - s3587630

## General Overview

#### Executables

main.py - the main executable

mainPreprocess.py - executable only for the preprocessing

mainSegment.py - executable only for the segmentation

#### Packages:

- Preprocess - module containing the preprocessing steps. Can also be used as a standalone application.

- Segment - module containing the segmentation of an image into lines and, further, segmentation of lines into words. The prerequisite for this step is that the image must be preprocessed.

- Classifier - module containing the classifier used for recognition of the handwriting text. This can also be used as a standalone application.

- Utils - contains common functions for handling reading/writing of data.

#### I/O packages:

- Input - contains the input data for preprocessing

- Output - contains the output data for preprocessing and segmentation

## Commands

(This application is programmed in Python 3. For brevity, `python3` commands will be used as `python`).

Installation:

- install Python 3 (ideally 3.6.x, but not 3.7.x because it does not support TensorFlow yet)

- install `pipenv` using `pip install pipenv`

- install the program using `pipenv install` when you are in the main folder.

For preprocessing:

- `pipenv run python Preprocess\preprocess.py name-of-input-file name-of-output-directory`

- or `pipenv run python mainPreprocess.py` -> the input file must be manually typed in.

For segmentation:

- `pipenv run python Segment\segment.py name-of-input-file name-of-output-directory`

- or `pipenv run python mainSegment.py` -> the input file must be manually typed in.

For Classifier:

- `pipenv run Classifier\train_network.py`

All together:
- `python main.py input_folder`

## Samples

1. Command line help:

   `pipenv run python main.py -h`

   Output:

  ```
  python main.py -h
  usage: main.py [-h] [-d | -f] [-e EXTENSION] input_dir

  positional arguments:
    input_dir             Specify the input directory

  optional arguments:
    -h, --help            show this help message and exit
    -d, --debug           run program in debug mode
    -f, --fast            skip over intermediary results (used to speed-up the
                          program)
    -e EXTENSION, --extension EXTENSION
                          specify the extension (default=.jpg)  
    -o OUTPUT, --output OUTPUT
                          specify the output directory
  ```
  
2. Sample running 3 images of about 2156 x 1625 in FAST mode:

   Using the command: `pipenv run python main.py -f InputDummy -o OutputDummy`

  ```
  pipenv run python main.py -f InputDummy -o OutputDummy

  Processing files from directory:   InputDummy
  With extension:                    *jpg
  Output Directory:                  OutputDummy
  Runmode:                           FAST


  Processing image1.jpg
  Finished preprocessing image1.jpg
      ****
  Segmenting image1.jpg
  Finished Segmenting image1.jpg

  Elapsed time: 9.210001230239868
  -----------------------------------

  Processing image2.jpg
  Finished preprocessing image2.jpg
      ****
  Segmenting image2.jpg
  Finished Segmenting image2.jpg

  Elapsed time: 9.017125844955444
  -----------------------------------

  Processing image3.jpg
  Finished preprocessing image3.jpg
      ****
  Segmenting image3.jpg
  Finished Segmenting image3.jpg

  Elapsed time: 9.497315168380737
  -----------------------------------
  Total elapsed time: 27.72841787338257
  Average/image = 9.241480747858683
  -----------------------------------
  ```
