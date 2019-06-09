import sys, os, shutil

'''
This file contains function for checking the existance of directories,
creating directories or remove directories recursively.
'''

def ensure_directory (directory):
    directory =  os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)

def remove_directory(directory):
    directory_path = os.path.abspath(directory)
    if os.path.exists(directory):
        shutil.rmtree(directory_path)
            
