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

(This application is programmed in Python 3. For brevity, `python3` commands will be used as `python` and `pip3` command will be used as `pip`). We have two methods for creating virtual environment (`virtualenv` or `pipenv`), choose anyone.

Installation:

- install Python 3 (ideally 3.6.x, but not 3.7.x because it does not support TensorFlow yet)

# 1 -Using virtualenv:

- install `virtualenv` using `pip install virtualenv`
- navigate to the main folder
- create virtual environment using `virtualenv -p python3 env`
- activate the environment using `source env/bin/activate`
- install all the required libraries using `pip install -r requirements.txt`


# 2 - Using pipenv:

- install `pipenv` using `pip install pipenv`
- install the program using `pipenv install` when you are in the main folder.

Notes about `pipenv`:
- If `pipenv` is used (or perhaps `virtualenv`), then `pipenv run` must be appended to the following commands.


### Running the full pipeline:

- `python main.py input_folder`

### For testing individually:

For preprocessing:

- `python mainPreprocess.py` -> the input file must be manually typed in.

For segmentation:

- `python mainSegment.py` -> the input file must be manually typed in.

For Classifier:

- `python Classifier\train_network.py` -> for training the network

- `python mainClassifier.py` -> for classification. Input file must be manually typed in.


## Samples

1. Command line help:

   when using `pipenv`:
   `pipenv run python main.py -h` 
   
   when using `virtualenv`:
   `python main.py -h`

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
  
2. Sample running 4 images of about 2156 x 1625 in FAST mode:

   Using the command: </br>
   `pipenv run python main.py -f InputTest -o OutputTest` -- incase of `pipenv` OR, </br>
   `python main.py -f InputTest -o OutputTest` -- incase of `virtualenv`
  
  Output:

  ```
  pipenv run python main.py -f InputTest -o OutputTest

  Processing files from directory:   InputTest
  With extension:                    *jpg
  Output Directory:                  OutputTest
  Runmode:                           NORMAL


  Processing image1.jpg
  Finished preprocessing image1.jpg
      ****
  Segmenting image1.jpg
  Finished segmenting image1.jpg
      ****
  Classifying image1.jpg
  Finished classifying image1.jpg

  Output is in: THIS-IS-PRIVATE-PATH\OutputTest\image1\output.txt

  Elapsed time: 10.896777629852295
  -----------------------------------

  Processing image2.jpg
  Finished preprocessing image2.jpg
      ****
  Segmenting image2.jpg
  Finished segmenting image2.jpg
      ****
  Classifying image2.jpg
  Finished classifying image2.jpg

  Output is in: THIS-IS-PRIVATE-PATH\OutputTest\image2\output.txt
  
  Elapsed time: 11.643365621566772
  -----------------------------------

  Processing image3.jpg
  Finished preprocessing image3.jpg
      ****
  Segmenting image3.jpg
  Finished segmenting image3.jpg
      ****
  Classifying image3.jpg
  Finished classifying image3.jpg

  Output is in: THIS-IS-PRIVATE-PATH\OutputTest\image3\output.txt

  Elapsed time: 11.344895839691162
  -----------------------------------

  Processing image4.jpg
  Finished preprocessing image4.jpg
      ****
  Segmenting image4.jpg
  Finished segmenting image4.jpg
      ****
  Classifying image4.jpg
  Finished classifying image4.jpg

  Output is in: THIS-IS-PRIVATE-PATH\OutputTest\image4\output.txt

  Elapsed time: 10.71731948852539
  -----------------------------------
  Total elapsed time: 44.60835003852844
  Average/image = 11.150589644908905
  -----------------------------------
  ```
