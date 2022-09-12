import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    Output will be corrupted if not set intermediate variable with larger sizes, they will be capped at 255 since
    they are np.uint8.
"""


kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])


def filter_high_boost(img, A):
    assert A >= 1

    img_s = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    img_hb = img * (A - 1) + img_s
    return img_hb, img_s


def filter_high_boost_m(img, A):
    assert A >= 1
    img = img.astype(np.float64)
    H, W = img.shape
    img_s = np.zeros_like(img)
    img_pad = np.pad(img, 1, 'reflect')

    for h in range(H):
        for w in range(W):
            img_s[h, w] = (img_pad[h: h + 3, w: w + 3] * kernel).sum()

    img_hb = img * (A - 1) + img_s
    return np.clip(np.round(img_hb), a_min=0, a_max=255).astype(np.uint8), \
           np.clip(np.round(img_s), a_min=0, a_max=255).astype(np.uint8)


def main():
    img = cv2.imread('flower.tif', flags=0)  # flags = 0 to read grayscale images

    _, img_s_cv = filter_high_boost(img, 2)

    img_hb_m1, img_s_m = filter_high_boost_m(img, 2)
    img_hb_m2, _ = filter_high_boost_m(img, 3)

    plt.figure(0)
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)
    axs[0, 0].imshow(img, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: flower.tif')
    axs[1, 0].imshow(img, cmap='Greys_r')
    axs[1, 0].set_title(f'Original image: flower.tif')

    axs[0, 1].imshow(img_s_cv, cmap='Greys_r')
    axs[0, 1].set_title(f'Laplacian filtered image: OpenCV')
    axs[1, 1].imshow(img_s_m, cmap='Greys_r')
    axs[1, 1].set_title(f'Laplacian filtered image: Manual')

    axs[0, 2].imshow(img_hb_m1, cmap='Greys_r')
    axs[0, 2].set_title(f'A = 2 | Manual')
    axs[1, 2].imshow(img_hb_m2, cmap='Greys_r')
    axs[1, 2].set_title(f'A = 3 | Manual')
    plt.show()


if __name__ == '__main__':
    main()
