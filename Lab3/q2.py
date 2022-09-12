import cv2
import matplotlib.pyplot as plt

import keyboard


def equalize_1(img):
    for c in range(3):
        img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return img


def equalize_2(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_new = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_new


def equalize_3(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_new = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_new


def main():
    img = cv2.imread('flowers.jpg')[:, :, ::-1]  # convert bgr to rgb

    res1 = equalize_1(img)
    res2 = equalize_2(img)
    res3 = equalize_3(img)

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
