import sys
import numpy as np
from tester_basemodel import Tester_basemodel
from tester_hybridmodel import Tester_hybridmodel
from argparse import ArgumentParser
from utils import downloadModels
import glob
import multiprocessing
import os

def decode_list(compressed_file_path, no_proc = 1):
    file_list = glob.glob(compressed_file_path)
    jobs = []
    idx = 0
    for filepath in file_list:
        process = multiprocessing.Process(target=decode, args=(filepath,))

        idx += 1
        print(idx)
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

    if idx % no_proc != 0:
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()


def decode(compressed_file_path):

    downloadModels()

    fileobj = open(compressed_file_path, mode='rb')
    buf = fileobj.read(1)
    arr = np.frombuffer(buf, dtype=np.uint8)
    b = int(arr[0])
    model_type = b % 2
    quality_level = b >> 1
    fileobj.close()

    if model_type == 0:
        if quality_level <= 5:
            model_dir = './models/MSEopt/Base_model/' + str(quality_level)
            tester = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSEopt/Hybrid_model/' + str(quality_level)
            tester = Tester_hybridmodel(model_dir, model_type, quality_level)
    elif model_type == 1:
        if quality_level <= 5:
            model_dir =  './models/MSSSIMopt/Base_model/' + str(quality_level)
            tester = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSSSIMopt/Hybrid_model/' + str(quality_level)
            tester = Tester_hybridmodel(model_dir, model_type, quality_level)

    base = os.path.splitext(compressed_file_path)[0]
    recon_path =  base + '.png'

    print("Reconstructing.. (Input: {}, Output: {}, Quality level: {})".format(compressed_file_path, recon_path,
                                                                            quality_level))
    tester.decode(compressed_file_path, recon_path)
    print("Reconstruction completed")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--compressed_file_path', type=str, dest="compressed_file_path", default='./examples/output.cmp', help="input compressed file path")
    parser.add_argument('--no_proc', type=int, dest="no_proc", default=1, help="Number of parallel threads")
    # parser.add_argument('--recon_path', type=str, dest="recon_path", default='./examples/recon_example.png', help="target reconstructed image path")

    args = parser.parse_args()
    decode_list(**vars(args))
