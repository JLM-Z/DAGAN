from tensorlayer.prepro import *
import numpy as np
import skimage.measure
from time import localtime, strftime
import logging
import tensorflow as tf
import os
import scipy.fftpack

def distort_img(x):
    x = (x + 1.) / 2.  # 映射至0-1
    x = flip_axis(x, axis=1, is_random=True)  # axis = 1, flip left and right ,random
    x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
    x = rotation(x, rg=10, is_random=True, fill_mode='constant')  # Rotate an image randomly or non-randomly
    x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')  # Shift an image randomly non-randomly.
    x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')  # Zoom in and out of a single image
    x = brightness(x, gamma=0.05, is_random=True)   # Change the brightness of a single image, randomly or non-randomly.
    x = x * 2 - 1   # 映射至0-1
    return x


def to_bad_img(x, mask):
    """
    :param x:
    :param mask:采样方式
    :return:
    """
    x = (x + 1.) / 2.
    fft = scipy.fftpack.fft2(x[:, :, 0])
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    x = np.abs(x)
    x = x * 2 - 1
    return x[:, :, np.newaxis]


def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs


def ssim(data):
    """
    :param data: tuple or list just have two elements
    :return: ssim of good data and bad data
    """
    x_good, x_bad = data
    x_good = np.squeeze(x_good)  # Remove single-dimensional entries from the shape of an array.
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def psnr(data):
    """
    :param data: tuple or list just have two elements
    :return: psnr of good data and bad data
    """
    x_good, x_bad = data
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def vgg_prepro(x):
    """
    :param x: An image with dimension of [row, col, channel] (default)
    :return: An image of resizing , tiling and regularization
    """
    x = imresize(x, [244, 244], interp='bilinear',
                 mode=None)  # resize the iamge.size to [244,244],rescale the value to [0, 255].
    x = np.tile(x, 3)  # 重复三次
    x = x / 127.5 - 1
    return x


def logging_setup(log_dir):
    """
    :param log_dir: str like , a director for store your logging
    :return: 1th - 3th are object of logger ,4th - 5th are the filename to store logging
    """
    current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())  # 获取当前时间点
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

    log_all = logging.getLogger('log_all')  #
    log_all.setLevel(logging.DEBUG)  # setLevel DEBUG 10
    log_all.addHandler(logging.FileHandler(log_all_filename))  # Open the specified file and use
    # it as the stream for logging.
    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)  # setLevel INFO 20
    log_eval.addHandler(logging.FileHandler(log_eval_filename))

    log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

    log_50 = logging.getLogger('log_50')
    log_50.setLevel(logging.DEBUG)
    log_50.addHandler(logging.FileHandler(log_50_filename))

    return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


if __name__ == "__main__":
    pass
