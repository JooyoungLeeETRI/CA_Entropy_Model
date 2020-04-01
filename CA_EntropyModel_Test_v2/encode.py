import sys
from tester_basemodel import Tester_basemodel
from tester_hybridmodel import Tester_hybridmodel
from argparse import ArgumentParser
from utils import downloadModels
import glob
import os
import numpy as np
import multiprocessing
from PIL import Image


def encode_list(model_type, input_path, quality_level, no_proc = 1):
    file_list = glob.glob(input_path)

    dir = './compressed_files/{}/{}'.format(model_type, quality_level)
    os.makedirs(dir, exist_ok=True)
    bpp_list = []
    filesize_list = []

    jobs = []
    idx = 0

    for filepath in file_list:
        print (str(idx+1) + ":" + filepath)
        compressed_file_path = '{}/{}.cmp'.format(dir, os.path.splitext(os.path.basename(filepath))[0])
        idx += 1
        # if not os.path.isfile(compressed_file_path):
        process = multiprocessing.Process(target=encode, args=(model_type, filepath, compressed_file_path, quality_level))
        if idx % no_proc != 0:
            jobs.append(process)
            # decode(filepath)
        elif idx % no_proc == 0:
            jobs.append(process)
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            jobs = []
        # encode(filepath, compressed_file_path, quality_level)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    for filepath in file_list:
        img = Image.open(filepath)
        w, h = img.size
        size = w * h

        compressed_file_path = '{}/{}.cmp'.format(dir, os.path.splitext(os.path.basename(filepath))[0])
        file_size = os.path.getsize(compressed_file_path) * 8
        bpp = file_size / size
        bpp_list.append(bpp)

        filesize_list.append(file_size)
    avg_bpp = np.mean(np.asarray(bpp_list))
    total_filesize = np.sum(np.asarray(filesize_list))
    bpp_list.append(avg_bpp)
    filesize_list.append(total_filesize)
    np.savetxt('{}/TEST_RESULT_BPP.csv'.format(dir), bpp_list, delimiter=",")
    np.savetxt('{}/TEST_RESULT_FILESIZE.csv'.format(dir), filesize_list, delimiter=",")


def encode(model_type, input_path, compressed_file_path, quality_level, no_proc=1):

    downloadModels()

    if quality_level not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(
            'Value of quality_grade option must be one of [1, 2, 3, 4, 5, 6, 7, 8, 9].')
        sys.exit(1)

    if model_type == 0:
        if quality_level <= 5:
            model_dir = './models/MSEopt/Base_model/' + str(quality_level)
            tester = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSEopt/Hybrid_model/' + str(quality_level)
            tester = Tester_hybridmodel(model_dir, model_type, quality_level)

    elif model_type == 1:
        if quality_level <= 5:
            model_dir = './models/MSSSIMopt/Base_model/' + str(quality_level)
            tester = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSSSIMopt/Hybrid_model/' + str(quality_level)
            tester = Tester_hybridmodel(model_dir, model_type, quality_level)
    else:
        print(
            'Model type parameter must be 0(MSE -optimized) or 1(MS-SSIM optimized).')
        sys.exit(1)

    print("Compressing.. (Input: {}, Quality level: {})".format(input_path, quality_level))
    tester.encode(model_type, input_path, compressed_file_path, quality_level)
    print("Compression completed")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=int, dest="model_type", default=0, choices=[0, 1], help="0: MSE optimized 1: MS-SSIM optimized" )
    parser.add_argument('--input_path', type=str, dest="input_path", default='./examples/input_example.png', help="input image path" )
    parser.add_argument('--quality_level', type=int, dest="quality_level", default=5, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="quality level from 1 to 9. The higher, te better")
    parser.add_argument('--no_proc', type=int, dest="no_proc", default=1, help="Number of parallel threads")

    args = parser.parse_args()
    encode_list(**vars(args))
