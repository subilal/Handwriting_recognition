# Group 7

Authors:

- Stefan Evanghelides - s2895323

- Huaitin Lu - s3339246

- Sudhakaran Jain - s3558487

- Name #4 - S#4

- Name #5 - S#5

# General Overview

main.py - the main executable

Preprocess - module containing the preprocessing steps. Can also be used as a standalone application.

Recognition - module containing the classifier used for recognition of the handwriting text. This can also be used as a standalone application.

# Commands

(This application is programmed in Python 3. For brevity, `python3` commands will be used as `python`).

Installation:

- install Python 3 (ideally 3.6.x, but not 3.7.x because it does not support TensorFlow yet)

- install `pipenv` using `pip install pipenv`

- install the program using `pipenv install` when you are in the main folder.

For preprocessing:

- `pipenv run python Preprocess\preprocess.py name-of-input-file name-of-output-file`

For recognition:

- `pipenv run Classifier\train_network.py`

All together:
- `python main.py input_folder`
