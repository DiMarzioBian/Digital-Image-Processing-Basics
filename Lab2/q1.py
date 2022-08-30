import cv2
import numpy as np
import matplotlib.pyplot as plt

import keyboard


def add_noise(img, level):
    bins = np.arange(1, level) / level * 256 - 1
    img_new = np.digitize(img, bins)
    return img_new


def filter_noise(img, level):
    bins = np.arange(1, level) / level * 256 - 1
    img_new = np.digitize(img, bins)
    return img_new


def main():
    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_m = cv2.imread('Mandrill.bmp', flags=0)
    img_p = cv2.imread('Peppers.bmp', flags=0)

    i = 0
    map_level = {0: 2, 1: 4, 2: 8, 3: 16, 4: 64}
    level = map_level[i]


    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_m = cv2.imread('Mandrill.bmp', flags=0)
    img_p = cv2.imread('Peppers.bmp', flags=0)

    print('Start histogram transferring...')
    img_l_new, hist_l, hist_l_new = histTransfer(img_l, img_m)
    img_m_new, hist_m, hist_m_new = histTransfer(img_m, img_p)
    img_p_new, hist_p, hist_p_new = histTransfer(img_p, img_l)

    print(f'Drawing histogram transferred images...')
    # lena to mandrill
    plt.figure(0)
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10, 10)
    fig.suptitle(f'Transferred "Lena.bmp" from histogram of "Mandrill.bmp".')
    axs[0, 0].imshow(img_l, cmap='Greys_r')
    axs[0, 1].plot(hist_l)
    axs[1, 0].imshow(img_m, cmap='Greys_r')
    axs[1, 1].plot(hist_m)
    axs[2, 0].imshow(img_l_new, cmap='Greys_r')
    axs[2, 1].plot(hist_l_new)
    plt.show()


if __name__ == '__main__':
    main()
