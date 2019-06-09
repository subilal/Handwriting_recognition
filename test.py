import sys, os

import argparse
import glob

from Preprocess.preprocess import *
from Segment.segment import *


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-d", "--debug", help="run program in debug mode",
                                    action="store_true")
group.add_argument("-f", "--fast",  help="skip over intermediary results; used to speed-up the program",
                                    action="store_true")
parser.add_argument("input",        help="Specify the input directory")
args = parser.parse_args()

debug = args.debug
fast = args.fast

print ('Debug =' + str(debug))
print ('Fast =' + str(fast))
print ('Input Directory =' + args.input)
