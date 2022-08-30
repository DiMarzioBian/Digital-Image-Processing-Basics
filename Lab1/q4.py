import cv2
import numpy as np
import matplotlib.pyplot as plt


def histTransfer(img_1, img_2):
    hist_2 = cv2.calcHist([img_2], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    return img_1


def main():
    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_m = cv2.imread('Mandrill.bmp', flags=0)
    img_p = cv2.imread('Peppers.bmp', flags=0)

    print('Start histogram transferring...')
    img_l_new = histTransfer(img_l, img_m)
    img_m_new = histTransfer(img_m, img_p)
    img_p_new = histTransfer(img_p, img_l)

    print(f'Drawing histogram transferred images...')
    # lena to mandrill
    plt.figure(0)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)
    fig.suptitle(f'Transferred "Lena.bmp" from histogram of "Mandrill.bmp".')
    axs[0].imshow(img_l, cmap='Greys_r')
    axs[1].imshow(img_m, cmap='Greys_r')
    axs[2].imshow(img_l_new, cmap='Greys_r')
    plt.show()

    # mandrill to peppers
    plt.figure(1)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)
    fig.suptitle(f'Transferred "Mandrill.bmp" from histogram of "Peppers.bmp".')
    axs[0].imshow(img_m, cmap='Greys_r')
    axs[1].imshow(img_p, cmap='Greys_r')
    axs[2].imshow(img_m_new, cmap='Greys_r')
    plt.show()

    # peppers to lena
    plt.figure(2)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)
    fig.suptitle(f'Transferred "Peppers.bmp" from histogram of "Lena.bmp".')
    axs[0].imshow(img_p, cmap='Greys_r')
    axs[1].imshow(img_l, cmap='Greys_r')
    axs[2].imshow(img_p_new, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    main()
