## Introduction
This code is the second version of the test code for evaluating a image compression model proposed in our work ["Context-adaptive Entropy Model for End-to-end Optimized Image Compression"](http://arxiv.org/abs/1809.10452). Now you can compress images of various sizes much more efficiently, **with THE SAME MODELS**. If you've already downloaded our previous models, just copy them into the "models" directory under the working directory of this code. A few options are different from the version 1. Please refer to the detailed instruction below:


## Supported I/O formats
Python Imaging Library (PIL) is used in our test code. To file the supported formats, please refer to https://pillow.readthedocs.io/en/4.1.x/handbook/image-file-formats.html. PNG and BMP formats including RGB channels have been tested.


### Encoding
* usage: 
encode.py [-h] [--model_type {0,1}] [--input_path INPUT_PATH] [--quality_level {1,2,3,4,5,6,7,8,9}] [--no_proc]

* optional arguments:
  	* --model_type {0,1}  
  	0: MSE optimized 1: MS-SSIM optimized
  	* --input_path  
  	input image path
		* *You can use a wildcard character in the filename.*
		* *The output files will be generated under "compressed_files" directory.*
  	* --quality_level {1,2,3,4,5,6,7,8,9}  
  	quality level from 1 to 9. The higher, the better
  	* --no_proc  
  	number of processes to work in parallel  
    &nbsp;
* Sample command for encoding:  
python encode.py --model_type 0 --input_path ./examples/*.png --quality_level 5 --no_proc 4  
* ***[Note] Some environments expand the asterisk to every matching file before passing the arguments. To deal with this issue, you can modify "encode.py" for your own system.*** 


### Decoding
* usage: 
decode.py [-h] [--compressed_file_path COMPRESSED_FILE_PATH]

* optional arguments:
  * --compressed_file_path 	input compressed file path
		* *You can use a wildcard character in the filename.*
		* *The output files will be generated in the same path as the input compressed files.*

* Sample command for decoding:  
python decode.py --compressed_file_path ./compressed_files/0/5/*.cmp --no_proc 4
* ***[Note] Some environments expand the asterisk to every matching file before passing the arguments. To deal with this issue, you can modify "decode.py" for your own system.***