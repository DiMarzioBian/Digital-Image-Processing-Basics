import cv2
import numpy as np
import matplotlib.pyplot as plt

import keyboard


def localEqualizeHist(img, len_window):
    len_one_side = len_window // 2
    img_new = np.ones_like(img) * -1
    img_pad = np.pad(img, len_one_side, 'reflect')

    for row in range(img_new.shape[0]):
        for col in range(img_new.shape[1]):
            sub_img = img_pad[row: (row + len_window), col: (col + len_window)]
            img_new[row, col] = cv2.equalizeHist(sub_img)[len_one_side, len_one_side]

    return img_new


def main():
    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_m = cv2.imread('Mandrill.bmp', flags=0)
    img_p = cv2.imread('Peppers.bmp', flags=0)

    i = 0
    map_level = {0: 3, 1: 21, 2: 41, 3: 61}
    level = map_level[i]

    while True:
        print('\nPress esc to terminate.')
        print('Press any key to continue.')
        if keyboard.read_key() != 'esc':
            print(f'Drawing local histogram equalization by window size {level} x {level}...\n')

            img_l_new = localEqualizeHist(img_l, level)
            img_m_new = localEqualizeHist(img_m, level)
            img_p_new = localEqualizeHist(img_p, level)

            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(10, 3)
            fig.suptitle(f'Local histogram equalized images by window size {level} x {level}.')
            axs[0].imshow(img_l_new, cmap='Greys_r')
            axs[1].imshow(img_m_new, cmap='Greys_r')
            axs[2].imshow(img_p_new, cmap='Greys_r')
            plt.show()
            plt.savefig(f'img/img_q3_{level}.png')
        else:
            break

        i += 1
        if i >= len(map_level):
            break
        else:
            level = map_level[i]


if __name__ == '__main__':
    main()
