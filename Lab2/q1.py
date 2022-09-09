import cv2
import numpy as np
import matplotlib.pyplot as plt

import keyboard


def add_noise(img, level):
    bins = np.arange(1, level) / level * 256 - 1
    img_new = np.digitize(img, bins)
    return img_new


def filter_median(img, level):
    bins = np.arange(1, level) / level * 256 - 1
    img_new = np.digitize(img, bins)
    return img_new


def filter_gaussian(img, level):
    bins = np.arange(1, level) / level * 256 - 1
    img_new = np.digitize(img, bins)
    return img_new


def main():
    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_f = cv2.imread('flower.tif', flags=0)

    img_ln = add_noise(img_l)
    img_fn = add_noise(img_f)

    img_lnm = filter_median(img_ln)
    img_lng = filter_gaussian(img_ln)

    img_fnm = filter_median(img_fn)
    img_fng = filter_gaussian(img_fn)

    # lena to mandrill
    plt.figure(0)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].plot(img_l)
    axs[0, 1].plot(img_ln)
    axs[1, 0].plot(img_lnm)
    axs[1, 1].plot(img_fnm)
    plt.show()

    # lena to mandrill
    plt.figure(1)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].plot(img_f)
    axs[0, 1].plot(img_fn)
    axs[1, 0].plot(img_lng)
    axs[1, 1].plot(img_fng)
    plt.show()


if __name__ == '__main__':
    main()
