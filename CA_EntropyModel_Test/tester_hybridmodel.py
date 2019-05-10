import tensorflow as tf
import arithmeticcoding
import numpy as np
import os
import math
from utils import printProgressBar

from PIL import Image

class Tester_hybridmodel(object):
    def __init__(self, model_dir, model_type, quality_level):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model_dir = model_dir
        self.M1 = 192
        if model_type==0: #MSE optimized
            if quality_level == 6:
                self.M2 = 192
            elif quality_level == 7:
                self.M2 = 228
            elif quality_level in [8,9]:
                self.M2 = 408
        elif model_type==1: #MS-SSIM optimized
            if quality_level in [6,7]:
                self.M2 = 192
            elif quality_level in [8,9]:
                self.M2 = 408
        self.M = self.M1 + self.M2
        self.build_model()

    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def build_model(self):
        frozen_graph_filename = self.model_dir + "/" + "saved_model.pb"
        graph = self.load_graph(frozen_graph_filename)

        # for verifying
        # for op in graph.get_operations():
        #     print(op.name)

        self.recon_image = graph.get_tensor_by_name('prefix/clip_by_value:0')
        self.y_hat = graph.get_tensor_by_name('prefix/transpose_3:0')
        self.input_x = graph.get_tensor_by_name('prefix/Placeholder_2:0')
        self.h_s_out = graph.get_tensor_by_name('prefix/transpose_8:0')
        self.z_hat = graph.get_tensor_by_name('prefix/Round_1:0')
        self.sigma_z = graph.get_tensor_by_name('prefix/H/Abs:0')
        self.pred_mean = graph.get_tensor_by_name('prefix/H/P_15/strided_slice:0')
        self.pred_sigma = graph.get_tensor_by_name('prefix/H/P_15/Exp:0')
        self.concatenated_c_i = graph.get_tensor_by_name('prefix/H/concat_45:0')

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(graph=graph, config=config)

    def extractor_prime(self, padded_c_prime, h_idx, w_idx):  # with TOP N dimensions
        return padded_c_prime[:, :, h_idx:h_idx + 4, w_idx:w_idx + 4]

    def extractor_doubleprime(self, padded_y_hat, h_idx, w_idx):  # with TOP N dimensions
        # Masking has no effect on decoding, because unknown variables are already set to zeros.
        # Therefore, the masking can be skipped in the case of decoding.
        # We just leave it here just to maintain a simple structure of code.
        mask = [[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0]]
        return np.multiply(padded_y_hat[:, :, h_idx:h_idx + 4, w_idx:w_idx + 4], mask)


    def encode(self, model_type, input_path, compressed_file_path, quality_level):  # with TOP N dimensions

        img = Image.open(input_path)
        w, h = img.size

        fileobj = open(compressed_file_path, mode='wb')

        buf = quality_level << 1
        buf = buf + model_type
        arr = np.array([0], dtype=np.uint8)
        arr[0] =  buf
        arr.tofile(fileobj)

        arr = np.array([w, h], dtype=np.uint16)
        arr.tofile(fileobj)
        fileobj.close()

        new_w = int(math.ceil(w / 64) * 64)
        new_h = int(math.ceil(h / 64) * 64)

        pad_w = new_w - w
        pad_h = new_h - h

        input_x = np.asarray(img)
        input_x = np.pad(input_x, ((pad_h,0), (pad_w,0), (0,0)), mode='reflect')
        input_x = input_x.reshape(1, new_h, new_w, 3)
        input_x = input_x.transpose([0, 3, 1, 2])

        h_s_out, y_hat, z_hat, sigma_z = self.sess.run([self.h_s_out, self.y_hat, self.z_hat, self.sigma_z],
                                                       feed_dict={self.input_x: input_x})  # NCHW

        ############### encode z ####################################
        bitout = arithmeticcoding.BitOutputStream(open(compressed_file_path, "ab+"))
        enc = arithmeticcoding.ArithmeticEncoder(bitout)

        printProgressBar(0, z_hat.shape[1], prefix='Encoding z_hat:', suffix='Complete', length=50)
        for ch_idx in range(z_hat.shape[1]):
            printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Encoding z_hat:', suffix='Complete', length=50)
            mu_val = 255
            sigma_val = sigma_z[ch_idx]
            # exp_sigma_val = np.exp(sigma_val)

            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            for h_idx in range(z_hat.shape[2]):
                for w_idx in range(z_hat.shape[3]):
                    symbol = np.rint(z_hat[0, ch_idx, h_idx, w_idx] + 255)
                    if symbol < 0 or symbol > 511:
                        print("symbol range error: " + str(symbol))

                    # print(symbol)
                    enc.write(freq, symbol)

        # enc.write(freq, 512)
        # enc.finish()
        # bitout.close()

        ############### encode y ####################################
        padded_y1_hat = np.pad(y_hat[:, :self.M1, :, :], ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                               constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        # bitout = arithmeticcoding.BitOutputStream(open(enc_outputfile, "wb"))
        # enc = arithmeticcoding.ArithmeticEncoder(bitout)

        c_prime = h_s_out[:, :self.M1, :, :]
        sigma2 = h_s_out[:, self.M1:, :, :]
        padded_c_prime = np.pad(c_prime, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        printProgressBar(0, y_hat.shape[2], prefix='Encoding y_hat:', suffix='Complete', length=50)
        for h_idx in range(y_hat.shape[2]):
            printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Encoding y_hat:', suffix='Complete', length=50)
            for w_idx in range(y_hat.shape[3]):
                c_prime_i = self.extractor_prime(padded_c_prime, h_idx, w_idx)
                c_doubleprime_i = self.extractor_doubleprime(padded_y1_hat, h_idx, w_idx)
                concatenated_c_i = np.concatenate([c_doubleprime_i, c_prime_i], axis=1)

                pred_mean, pred_sigma = self.sess.run([self.pred_mean, self.pred_sigma],
                                                      feed_dict={self.concatenated_c_i: concatenated_c_i})

                zero_means = np.zeros(
                    [pred_mean.shape[0], self.M2, pred_mean.shape[2],
                     pred_mean.shape[3]])

                concat_pred_mean = np.concatenate([pred_mean, zero_means], axis=1)
                concat_pred_sigma = np.concatenate([pred_sigma, sigma2[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]], axis=1)

                for ch_idx in range(self.M):
                    mu_val = concat_pred_mean[0, ch_idx, 0, 0] + 255
                    sigma_val = concat_pred_sigma[0, ch_idx, 0, 0]
                    # exp_sigma_val = np.exp(sigma_val)

                    freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

                    symbol = np.rint(y_hat[0, ch_idx, h_idx, w_idx] + 255)
                    if symbol < 0 or symbol > 511:
                        print("symbol range error: " + str(symbol))
                    enc.write(freq, symbol)
        enc.write(freq, 512)
        enc.finish()
        bitout.close()

        return compressed_file_path


    def decode(self, compressed_file, recon_path):  # with TOP N dimensions

        fileobj = open(compressed_file, mode='rb')
        fileobj.read(1) #dummy
        buf = fileobj.read(4)
        arr = np.frombuffer(buf, dtype=np.uint16)
        w = int(arr[0])
        h = int(arr[1])


        padded_w = int(math.ceil(w / 64) * 64)
        padded_h = int(math.ceil(h / 64) * 64)

        y_hat, z_hat, sigma_z = self.sess.run([self.y_hat, self.z_hat, self.sigma_z],
                                              feed_dict={
                                                  self.input_x: np.zeros((1, 3, padded_h, padded_w))})  # NCHW

        padded_y1_hat = np.pad(y_hat[:, :self.M1, :, :], ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                               constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        ############### decode zhat ####################################
        bitin = arithmeticcoding.BitInputStream(fileobj)
        dec = arithmeticcoding.ArithmeticDecoder(bitin)

        printProgressBar(0, z_hat.shape[1], prefix='Decoding z_hat:', suffix='Complete', length=50)
        for ch_idx in range(z_hat.shape[1]):
            printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Decoding z_hat:', suffix='Complete', length=50)
            mu_val = 255
            sigma_val = sigma_z[ch_idx]
            # exp_sigma_val = np.exp(sigma_val)

            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            for h_idx in range(z_hat.shape[2]):
                for w_idx in range(z_hat.shape[3]):
                    symbol = dec.read(freq)
                    if symbol == 512:  # EOF symbol
                        print("EOF symbol")
                        break
                    z_hat[:, ch_idx, h_idx, w_idx] = symbol - 255

        # bitin.close()

        ##################
        #################################################
        # Entropy decoding y
        # padded_z = np.zeros_like(padded_z, dtype = np.float32)
        h_s_out = self.sess.run(self.h_s_out, feed_dict={self.z_hat: z_hat})
        c_prime = h_s_out[:, :self.M1, :, :]
        sigma2 = h_s_out[:, self.M1:, :, :]
        padded_c_prime = np.pad(c_prime, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        padded_y1_hat[:, :, :, :] = 0.0
        y_hat[:, :, :, :] = 0.0

        # bitin = arithmeticcoding.BitInputStream(open(dec_inputfile, "rb"))
        # dec = arithmeticcoding.ArithmeticDecoder(bitin)

        printProgressBar(0, y_hat.shape[2], prefix='Decoding y_hat:', suffix='Complete', length=50)
        for h_idx in range(y_hat.shape[2]):
            printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Decoding y_hat:', suffix='Complete', length=50)
            for w_idx in range(y_hat.shape[3]):
                c_prime_i = self.extractor_prime(padded_c_prime, h_idx, w_idx)
                c_doubleprime_i = self.extractor_doubleprime(padded_y1_hat, h_idx, w_idx)
                concatenated_c_i = np.concatenate([c_doubleprime_i, c_prime_i], axis=1)

                pred_mean, pred_sigma = self.sess.run([self.pred_mean, self.pred_sigma],
                                                      feed_dict={self.concatenated_c_i: concatenated_c_i})

                zero_means = np.zeros(
                    [pred_mean.shape[0], self.M2, pred_mean.shape[2],
                     pred_mean.shape[3]])

                concat_pred_mean = np.concatenate([pred_mean, zero_means], axis=1)
                concat_pred_sigma = np.concatenate([pred_sigma, sigma2[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]], axis=1)

                for ch_idx in range(self.M):
                    mu_val = concat_pred_mean[0, ch_idx, 0, 0] + 255
                    sigma_val = concat_pred_sigma[0, ch_idx, 0, 0]

                    freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

                    symbol = dec.read(freq)
                    if symbol == 512:  # EOF symbol
                        print("EOF symbol")
                        break
                    if ch_idx < self.M1:
                        padded_y1_hat[:, ch_idx, h_idx + 3, w_idx + 2] = symbol - 255
                    y_hat[:, ch_idx, h_idx, w_idx] = symbol - 255
        bitin.close()

        #################################################

        recon = self.sess.run(self.recon_image, {self.y_hat: y_hat})
        recon = recon[0, -h:, -w:, :]

        im = Image.fromarray(recon.astype(np.uint8))
        im.save(recon_path)

        return
