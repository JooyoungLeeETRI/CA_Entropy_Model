import sys
from tester_basemodel import Tester_basemodel
from tester_hybridmodel import Tester_hybridmodel
from argparse import ArgumentParser
from utils import downloadModels

def encode(model_type, input_path, compressed_file_path, quality_level):

    downloadModels()

    if quality_level not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(
            'Value of quality_grade option must be one of [1, 2, 3, 4, 5, 6, 7, 8, 9]. High quality options, from 6 to 9 will be added soon')
        sys.exit(1)

    if model_type == 0:
        if quality_level <= 5:
            model_dir = './models/MSEopt/Base_model/' + str(quality_level)
            trainer = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSEopt/Hybrid_model/' + str(quality_level)
            trainer = Tester_hybridmodel(model_dir, model_type, quality_level)

    elif model_type == 1:
        if quality_level <= 5:
            model_dir = './models/MSSSIMopt/Base_model/' + str(quality_level)
            trainer = Tester_basemodel(model_dir)
        else:
            model_dir = './models/MSSSIMopt/Hybrid_model/' + str(quality_level)
            trainer = Tester_hybridmodel(model_dir, model_type, quality_level)
    else:
        print(
            'Model type parameter must be 1(MSE -optimized) or 2(MS-SSIM optimized).')
        sys.exit(1)

    print("Compressing.. (Input: {}, Output: {}, Quality level: {})".format(input_path, compressed_file_path, quality_level))
    trainer.encode(model_type, input_path, compressed_file_path, quality_level)
    print("Compression completed")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=int, dest="model_type", default=0, choices=[0, 1], help="0: MSE optimized 1: MS-SSIM optimized" )
    parser.add_argument('--input_path', type=str, dest="input_path", default='./examples/input_example.png', help="input image path" )
    parser.add_argument('--compressed_file_path', type=str, dest="compressed_file_path", default='./examples/output.cmp', help="target compressed file path" )
    parser.add_argument('--quality_level', type=int, dest="quality_level", default=5, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="quality level from 1 to 9. The higher, te better")

    args = parser.parse_args()
    encode(**vars(args))
