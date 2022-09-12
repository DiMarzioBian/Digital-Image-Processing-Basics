import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
"""


def rotate_hsv_clockwise(img, angle):
    img = img.astype(np.float)
    img[:, :, 0] += angle / 2
    img[:, :, 0][img[:, :, 0] >= 180] -= 180
    return img.astype(np.uint8)


def rgb_to_hsv(img):
    img = img.astype(np.float)
    Height, Width, Channels = img.shape
    img_hsv = np.zeros_like(img, dtype=np.float)

    for height in range(Height):
        for width in range(Width):
            (r, g, b) = img[height, width, :]

            c_max = np.max([r, g, b])
            c_min = np.min([r, g, b])
            c_range = (c_max-c_min)

            if c_min == c_max:
                img_hsv[height, width, :] = [0, 0, c_max]
                continue

            rc = (c_max - r) * 60 / c_range
            gc = (c_max - g) * 60 / c_range
            bc = (c_max - b) * 60 / c_range
            if r == c_max:
                h = bc - gc
            elif g == c_max:
                h = 120 + rc - bc
            else:
                h = 240 + gc - rc

            h += 360 if h < 0 else 0
            img_hsv[height, width, :] = [h / 2, (c_max - c_min) * 255 / c_max, c_max]

    return np.floor(img_hsv).astype(np.uint8)


def hsv_to_bgr(img):
    img = img.astype(np.float)
    Height, Width, Channels = img.shape
    img_rgb = np.zeros_like(img, dtype=np.float)

    for height in range(Height):
        for width in range(Width):
            (h, s, v) = img[height, width, :]

            h60 = h / 60.0
            h60f = np.floor(h60)
            hi = int(h60f) % 6
            f = h60 - h60f
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            r, g, b = 0, 0, 0
            if hi == 0:
                r, g, b = v, t, p
            elif hi == 1:
                r, g, b = q, v, p
            elif hi == 2:
                r, g, b = p, v, t
            elif hi == 3:
                r, g, b = p, q, v
            elif hi == 4:
                r, g, b = t, p, v
            elif hi == 5:
                r, g, b = v, p, q
            img_rgb[height, width, :] = [r, g, b]

    return np.floor(img_rgb * 255).astype(np.uint8)


def main():
    img = cv2.imread('flowers.jpg')[:, :, ::-1]  # convert bgr to rgb
    mask = np.clip(np.repeat(np.expand_dims(cv2.imread('mask.tif', flags=0), axis=-1), 3, axis=-1), a_min=0, a_max=1)

    img_hsv = rgb_to_hsv(img)

    img_hsv_r1 = rotate_hsv_clockwise(img_hsv, 120)
    img_hsv_r2 = rotate_hsv_clockwise(img_hsv, 240)

    img_r1 = hsv_to_bgr(img_hsv_r1)
    img_r2 = hsv_to_bgr(img_hsv_r2)

    plt.figure(0)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)

    axs[0, 0].imshow(img * mask)
    axs[0, 0].set_title(f'Masked image: flowers.jpg')

    axs[0, 1].imshow(img[:, :, ::-1] * mask)
    axs[0, 1].set_title(f'Image swapped R and B')

    axs[1, 0].imshow(img_r1 * mask)
    axs[1, 0].set_title(f'H rotate | angle=120')

    axs[1, 1].imshow(img_r2 * mask)
    axs[1, 1].set_title(f'H rotate | angle=240')
    plt.show()


if __name__ == '__main__':
    main()
