import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
    HSV: V stands for value or brightness in HSB, dark area becomes darker since image is 
         pretty bright. Check middle bottom leaves.
    LAB: L stands for lightness, after equalized, dark area becomes darker since image is 
         pretty bright. Check right bottom corner.
"""
map_rgb_to_xyz = np.array([
    [0.412453, 0.357580, 0.1180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227]
])

Xn = 0.950456
Zn = 1.088754


def equalize_1(img_in):
    img = img_in.copy()
    for c in range(3):
        img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return img


def equalize_2(img_in):
    img = img_in.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_new = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_new


def equalize_3(img_in):
    img = img_in.copy()
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 0] = cv2.equalizeHist(img_lab[:, :, 0])
    img_new = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
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
