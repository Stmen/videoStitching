import cv2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#work_dir = os.getcwd()
base_dir = ""
data_dir = base_dir + "/data"
data_dir_fingerpoint = data_dir + "/fingerpoint"
data_dir_football = data_dir + "/football"

def test():
        commond = "python ./pano.py files3.txt"
        os.system(commond)
def main( args):
    test()


if __name__ == "__main__" :
    main(sys.argv[ 1:])



