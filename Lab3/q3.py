import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def transform_hough(img):
    for c in range(3):
        img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return img


def main():
    img = cv2.imread('sniper.jpg')
    img_hough = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold1, threshold2 = 60, 500
    edges = cv2.Canny(img_gray, threshold1=threshold1, threshold2=threshold2)
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_edges_hough = img_edges.copy()

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=5, minLineLength=70, maxLineGap=20)
    # lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=10)

    k_top = 10

    (x1, y1, x2, y2) = lines[0][0]
    cv2.line(img_hough, pt1=(x1, y1), pt2=(x2, y2), color=(0, 140, 255), thickness=5)
    cv2.line(img_edges_hough, pt1=(x1, y1), pt2=(x2, y2), color=(0, 140, 255), thickness=5)

    for line in lines[1:k_top]:
        (x1, y1, x2, y2) = line[0]
        cv2.line(img_hough, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
        cv2.line(img_edges_hough, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    plt.figure(0)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].imshow(img[:, :, ::-1])
    axs[0, 0].set_title(f'Original image: sniper.jpg')

    axs[0, 1].imshow(img_edges, cmap='Greys_r')
    axs[0, 1].set_title(f'Canny detected edges | ({threshold1}, {threshold2})')

    axs[1, 0].imshow(img_hough[:, :, ::-1])
    axs[1, 0].set_title(f'Top {k_top} prominent edge + image')

    axs[1, 1].imshow(img_edges_hough[:, :, ::-1])
    axs[1, 1].set_title(f'Top {k_top} prominent edges  + edges')

    plt.show()


if __name__ == '__main__':
    main()
