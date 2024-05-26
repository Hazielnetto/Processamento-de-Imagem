import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

param_1 = 20
param_2 = 31

def segmentarIris(img):
    """
    Segmenta a íris em uma imagem.

    Args:
        img (numpy.ndarray): Imagem de entrada (colorida).

    Returns:
        numpy.ndarray: Imagem da íris segmentada.
    """
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    _, threshold = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

    circles = None
    min_radius = 80
    max_radius = 200

    while circles is None or circles.size == 0:
        circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,
                                   2, 2000,
                                   param1=param_1,
                                   param2=param_2,
                                   minRadius=min_radius,
                                   maxRadius=max_radius)
        max_radius += 10

    circles = np.uint16(np.around(circles))
    mascara = np.zeros_like(img)

    for i in circles[0, :]:
        cv2.circle(mascara, (i[0], i[1]), i[2], (255, 255, 255), -1)

    return cv2.bitwise_and(img, mascara)

def removerPupila(img):
    """
    Remove a pupila da imagem.

    Args:
        img (numpy.ndarray): Imagem de entrada (colorida).

    Returns:
        numpy.ndarray: Imagem sem a pupila.
    """
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    _, threshold = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,
                               2, 500,
                               param1=param_1,
                               param2=param_2,
                               minRadius=20,
                               maxRadius=50)

    circles = np.uint16(np.around(circles))
    mascara = np.full_like(img, 255)

    for i in circles[0, :]:
        cv2.circle(mascara, (i[0], i[1]), i[2], (0, 0, 0), -1)

    return cv2.bitwise_and(img, mascara)

if __name__ == '__main__':
    for imgFile in os.listdir('HOUGH HAARCASCADE E SIFT/iris/saída'):
        os.remove(os.path.join('HOUGH HAARCASCADE E SIFT/iris/saída', imgFile))

    for fileName in os.listdir('HOUGH HAARCASCADE E SIFT/iris/imagens'):
        img = cv2.imread(os.path.join('HOUGH HAARCASCADE E SIFT/iris/imagens', fileName))
        iris = segmentarIris(img)
        irisSemPupila = removerPupila(iris)

        plt.imshow(irisSemPupila)
        plt.axis('off')
        plt.savefig(os.path.join('HOUGH HAARCASCADE E SIFT/iris/saída', fileName), format='jpg', dpi=300)
