#  Context-adaptive Entropy Model for End-to-end Optimized Image Compression
Repository of the paper "Context-adaptive Entropy Model for End-to-end Optimized Image Compression"

## Introduction
This repository includes the evaluation results and the reconstructed images of our paper "Context-adaptive Entropy Model for End-to-end Optimized Image Compression"(). Please refer to our paper() for the detailed information.

## Evaluation results
Rate-distortion curves of the proposed method and competitive methods. The top plot represents PSNR values in consequence of bpp changes, while the bottom plot shows MS-SSIM values in the same manner. MS-SSIM values are converted to decibels for differentiating the quality levels. Note we trained our model with two different types of distortion terms, MSE and MS-SSIM, and each of those two models outperforms all the other methods including [BPG](https://bellard.org/bpg/) for its target metric.

![Samples](./figures/samplecomparison.png)


![RD-PSNR](./figures/RD_PSNR.png)


![RD-MS-SSIM](./figures/RD_MSSSIM.png)

the compression gains in terms of BD-rate of PSNR over [JPEG2000](http://www.openjpeg.org/), [Balle'18(MSE-optimized)](https://arxiv.org/abs/1802.01436), [BPG](https://bellard.org/bpg/) are 34.08\%, 11.87\%, 6.85\%, respectively. In case of MS-SSIM, we found the wider gaps of 68.82\%, 13.93\%, 49.68\%, respectively.
