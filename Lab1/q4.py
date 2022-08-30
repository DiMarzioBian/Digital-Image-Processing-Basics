import cv2
import numpy as np
import matplotlib.pyplot as plt


def histTransfer(img_1, img_2):
    hist_1 = cv2.calcHist([img_1], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    hist_2 = cv2.calcHist([img_2], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    img_1_new = cv2.equalizeHist(img_1, hist_2)
    hist_1_new = cv2.calcHist([img_2], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    return img_1_new, hist_1, hist_1_new


def main():
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

    # mandrill to peppers
    plt.figure(1)
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10, 10)
    fig.suptitle(f'Transferred "Mandrill.bmp" from histogram of "Peppers.bmp".')
    axs[0, 0].imshow(img_m, cmap='Greys_r')
    axs[0, 1].plot(hist_m)
    axs[1, 0].imshow(img_p, cmap='Greys_r')
    axs[1, 1].plot(hist_p)
    axs[2, 0].imshow(img_m_new, cmap='Greys_r')
    axs[2, 1].plot(hist_m_new)
    plt.show()

    # peppers to lena
    plt.figure(2)
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10, 10)
    fig.suptitle(f'Transferred "Peppers.bmp" from histogram of "Lena.bmp".')
    axs[0, 0].imshow(img_p, cmap='Greys_r')
    axs[0, 1].plot(hist_p)
    axs[1, 0].imshow(img_l, cmap='Greys_r')
    axs[1, 1].plot(hist_l)
    axs[2, 0].imshow(img_p_new, cmap='Greys_r')
    axs[2, 1].plot(hist_p_new)
    plt.show()


if __name__ == '__main__':
    main()
