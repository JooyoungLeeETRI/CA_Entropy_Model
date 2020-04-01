import tensorflow as tf
import arithmeticcoding
import numpy as np
import os
import math
from utils import printProgressBar
import time

from PIL import Image

class Tester_basemodel(object):
    def __init__(self, model_dir):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.model_dir = model_dir
        self.M = 192
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

        # for op in graph.get_operations():
        #     print(op.name)
        # time.sleep(3)


        # for verifying
        # for op in graph.get_operations():
        #     print(op.name)

        self.recon_image = graph.get_tensor_by_name('prefix/clip_by_value:0')
        self.y_hat = graph.get_tensor_by_name('prefix/transpose_3:0')
        self.input_x = graph.get_tensor_by_name('prefix/Placeholder_2:0')
        self.c_prime = graph.get_tensor_by_name('prefix/transpose_8:0')
        self.z_hat = graph.get_tensor_by_name('prefix/Round_1:0')
        self.sigma_z = graph.get_tensor_by_name('prefix/H/Abs:0')
        self.pred_mean = graph.get_tensor_by_name('prefix/H/P_31/strided_slice:0')
        self.pred_sigma = graph.get_tensor_by_name('prefix/H/P_31/Exp:0')
        self.concatenated_c_i = graph.get_tensor_by_name('prefix/H/concat_31:0')

        self.gah1 = graph.get_tensor_by_name('prefix/E/gdn/truediv:0')
        self.gah2 = graph.get_tensor_by_name('prefix/E/gdn_2/truediv:0')
        self.gah3 = graph.get_tensor_by_name('prefix/E/gdn_3/truediv:0')

        self.gsh1 = graph.get_tensor_by_name('prefix/D/gdn/mul:0')
        self.gsh2 = graph.get_tensor_by_name('prefix/D/gdn_2/mul:0')
        self.gsh3 = graph.get_tensor_by_name('prefix/D/gdn_3/mul:0')


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

        new_w = int(math.ceil(float(w) / 2.0) * 2)
        new_h = int(math.ceil(float(h) / 2.0) * 2)

        pad_w = new_w - w
        pad_h = new_h - h

        img_array = np.asarray(img)
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        input_x = img_array

        input_x = input_x.reshape(1, new_h, new_w, 3)
        input_x = input_x.transpose([0, 3, 1, 2])

        gah1, sigma_z = self.sess.run(
            [self.gah1, self.sigma_z],
            feed_dict={self.input_x: input_x})

        gah1 = np.pad(gah1, ((0, 0), (0, gah1.shape[1] % 2), (0, gah1.shape[2] % 2), (0, 0)), mode='constant')
        gah2 = self.sess.run(
            self.gah2,
            feed_dict={self.gah1: gah1})

        gah2 = np.pad(gah2, ((0, 0), (0, gah2.shape[1] % 2), (0, gah2.shape[2] % 2), (0, 0)), mode='constant')
        gah3 = self.sess.run(
            self.gah3,
            feed_dict={self.gah2: gah2})

        gah3 = np.pad(gah3, ((0, 0), (0, gah3.shape[1] % 2), (0, gah3.shape[2] % 2), (0, 0)), mode='constant')
        y_hat = self.sess.run(
            self.y_hat,
            feed_dict={self.gah3: gah3})

        y_w = y_hat.shape[3]
        y_h = y_hat.shape[2]
        new_y_w = int(math.ceil(float(y_w) / 4.0) * 4)
        new_y_h = int(math.ceil(float(y_h) / 4.0) * 4)
        pad_y_w = new_y_w - y_w
        pad_y_h = new_y_h - y_h
        pad_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (0, pad_y_h), (0, pad_y_w)), mode='symmetric')
        z_hat, c_prime = self.sess.run(
            [self.z_hat, self.c_prime], feed_dict={self.y_hat: pad_y_hat})


        ############### encode zhat ####################################
        printProgressBar(0, z_hat.shape[1], prefix='Encoding z_hat:', suffix='Complete', length=50)
        bitout = arithmeticcoding.BitOutputStream(open(compressed_file_path, "ab+"))
        enc = arithmeticcoding.ArithmeticEncoder(bitout)

        for ch_idx in range(z_hat.shape[1]):
            printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Encoding z_hat:', suffix='Complete', length=50)
            mu_val = 255
            sigma_val = sigma_z[ch_idx]

            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            for h_idx in range(z_hat.shape[2]):
                for w_idx in range(z_hat.shape[3]):
                    symbol = np.int(z_hat[0, ch_idx, h_idx, w_idx] + 255)
                    if symbol < 0 or symbol > 511:
                        print("symbol range error: " + str(symbol))
                    enc.write(freq, symbol)

        ############### encode yhat ####################################
        padded_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                              constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        padded_c_prime = np.pad(c_prime, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        printProgressBar(0, y_hat.shape[2], prefix='Encoding y_hat:', suffix='Complete', length=50)
        for h_idx in range(y_hat.shape[2]):
            printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Encoding y_hat:', suffix='Complete', length=50)
            for w_idx in range(y_hat.shape[3]):


                c_prime_i = self.extractor_prime(padded_c_prime, h_idx, w_idx)
                c_doubleprime_i = self.extractor_doubleprime(padded_y_hat, h_idx, w_idx)
                concatenated_c_i = np.concatenate([c_doubleprime_i, c_prime_i], axis=1)

                pred_mean, pred_sigma = self.sess.run([self.pred_mean, self.pred_sigma],
                                                      feed_dict={self.concatenated_c_i: concatenated_c_i})

                for ch_idx in range(self.M):
                    mu_val = pred_mean[0, ch_idx, 0, 0] + 255
                    sigma_val = pred_sigma[0, ch_idx, 0, 0]

                    freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

                    symbol = np.int(y_hat[0, ch_idx, h_idx, w_idx] + 255)
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

        new_w = int(math.ceil(float(w) / 2.0) * 2)
        new_h = int(math.ceil(float(h) / 2.0) * 2)

        pad_w_1 = int((float(new_w) / 2.0) % 2)
        pad_h_1 = int((float(new_h) / 2.0) % 2)
        res_w_1 = math.floor(float(new_w) / 2.0) + pad_w_1
        res_h_1 = math.floor(float(new_h) / 2.0) + pad_h_1

        pad_w_2 = int((float(res_w_1) / 2.0) % 2)
        pad_h_2 = int((float(res_h_1) / 2.0) % 2)
        res_w_2 = math.floor(float(res_w_1) / 2.0) + pad_w_2
        res_h_2 = math.floor(float(res_h_1) / 2.0) + pad_h_2

        pad_w_3 = int((float(res_w_2) / 2.0) % 2)
        pad_h_3 = int((float(res_h_2) / 2.0) % 2)
        res_w_3 = math.floor(float(res_w_2) / 2.0) + pad_w_3
        res_h_3 = math.floor(float(res_h_2) / 2.0) + pad_h_3

        pad_w = new_w - w
        pad_h = new_h - h

        sigma_z = self.sess.run(self.sigma_z)
        y_hat = np.zeros((1, self.M, int(float(res_h_3) / 2.0), int(float(res_w_3) / 2.0)), dtype=np.float32)
        y_w = y_hat.shape[3]
        y_h = y_hat.shape[2]
        new_y_w = int(math.ceil(float(y_w) / 4.0) * 4)
        new_y_h = int(math.ceil(float(y_h) / 4.0) * 4)
        pad_y_w = new_y_w - y_w
        pad_y_h = new_y_h - y_h
        pad_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (0, pad_y_h), (0, pad_y_w)), mode='edge')
        z_hat = self.sess.run(self.z_hat, feed_dict={self.y_hat: pad_y_hat})  # NCHW

        ############### decode zhat ####################################
        bitin = arithmeticcoding.BitInputStream(fileobj)
        dec = arithmeticcoding.ArithmeticDecoder(bitin)

        z_hat[:, :, :, :] = 0.0

        printProgressBar(0, z_hat.shape[1], prefix='Decoding z_hat:', suffix='Complete', length=50)
        for ch_idx in range(z_hat.shape[1]):
            printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Decoding z_hat:', suffix='Complete', length=50)
            mu_val = 255
            sigma_val = sigma_z[ch_idx]

            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            for h_idx in range(z_hat.shape[2]):
                for w_idx in range(z_hat.shape[3]):
                    symbol = dec.read(freq)
                    if symbol == 512:  # EOF symbol
                        print("EOF symbol")
                        break
                    z_hat[:, ch_idx, h_idx, w_idx] = symbol - 255


        ############### decode yhat ####################################
        c_prime = self.sess.run(self.c_prime, feed_dict={self.z_hat: z_hat})
        # c_prime = np.round(c_prime, decimals=4)

        padded_c_prime = np.pad(c_prime, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        padded_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
                              constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
        padded_y_hat[:, :, :, :] = 0.0

        printProgressBar(0, y_hat.shape[2], prefix='Decoding y_hat:', suffix='Complete', length=50)
        for h_idx in range(y_hat.shape[2]):
            printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Decoding y_hat:', suffix='Complete', length=50)
            for w_idx in range(y_hat.shape[3]):
                c_prime_i = self.extractor_prime(padded_c_prime, h_idx, w_idx)
                c_doubleprime_i = self.extractor_doubleprime(padded_y_hat, h_idx, w_idx)
                concatenated_c_i = np.concatenate([c_doubleprime_i, c_prime_i], axis=1)

                pred_mean, pred_sigma = self.sess.run(
                    [self.pred_mean, self.pred_sigma],
                    feed_dict={self.concatenated_c_i: concatenated_c_i})

                for ch_idx in range(self.M):
                    mu_val = pred_mean[0, ch_idx, 0, 0] + 255
                    sigma_val = pred_sigma[0, ch_idx, 0, 0]

                    freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

                    symbol = dec.read(freq)
                    if symbol == 512:  # EOF symbol
                        print("EOF symbol")
                        break
                    padded_y_hat[:, ch_idx, h_idx + 3, w_idx + 2] = symbol - 255

        bitin.close()
        y_hat = padded_y_hat[:, :, 3:, 2:-1]
        #################################################

        ###############
        gsh1 = self.sess.run(self.gsh1, feed_dict={self.y_hat: y_hat})
        gsh1 = gsh1[:, :res_h_3 - pad_h_3, :res_w_3 - pad_w_3, :]

        gsh2 = self.sess.run(self.gsh2, feed_dict={self.gsh1: gsh1})
        gsh2 = gsh2[:, :res_h_2 - pad_h_2, :res_w_2 - pad_w_2, :]

        gsh3 = self.sess.run(self.gsh3, feed_dict={self.gsh2: gsh2})
        gsh3 = gsh3[:, :res_h_1 - pad_h_1, :res_w_1 - pad_w_1, :]

        recon = self.sess.run(self.recon_image, feed_dict={self.gsh3: gsh3})
        recon = recon[0, :recon.shape[1] - pad_h, :recon.shape[2] - pad_w, :]
        ###############

        im = Image.fromarray(recon.astype(np.uint8))
        im.save(recon_path)

        return
