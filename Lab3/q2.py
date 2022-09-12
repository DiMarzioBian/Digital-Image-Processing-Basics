import cv2
import matplotlib.pyplot as plt

import keyboard


def q1(img_bgr):
    img_rgb = img_bgr[:, :, (1, 0)]
    return img_rgb


def q2(img):
    img[:, :, (0, 1)] = img[:, :, (1, 0)]
    return img


def q3(img):
    img[:, :, (0, 1)] = img[:, :, (1, 0)]
    return img


def main():
    img = cv2.imread('flowers.tif', flags=0)  # flags = 0 to read grayscale images

    res1 = q1(img)
    res2 = q2(img)
    res3 = q3(img)

    plt.figure(0)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].imshow(img, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: flowers.tif')

    axs[0, 1].imshow(res1, cmap='Greys_r')
    axs[0, 1].set_title(f'Equalize individually')

    axs[1, 0].imshow(res2, cmap='Greys_r')
    axs[1, 0].set_title(f'Equalize V')

    axs[1, 1].imshow(res3, cmap='Greys_r')
    axs[1, 1].set_title(f'Equalize L')
    plt.show()


if __name__ == '__main__':
    main()
