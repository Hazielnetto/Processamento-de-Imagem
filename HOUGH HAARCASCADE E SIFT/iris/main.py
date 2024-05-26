#Alunos: Haziel Netto, Pedro Schumann e Vinícius Zoz

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

def segmentarIris(img):
    """
    Segmenta a íris em uma imagem.

    Argumentos:
        img: Imagem de entrada colorida.

    Retorna:
        imgProcessada: Imagem da íris segmentada.
    """
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    _, threshold = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

    max_radius = 200
    circles = 0
    while circles is 0 or circles.size == 0:
        circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,
                                   2, 2000, 
                                   param1=1, param2=1,
                                   minRadius=80, maxRadius=max_radius)
        max_radius += 10

    circles = np.uint16(np.around(circles))
    mascara = np.zeros_like(img)

    for i in circles[0, :]:
        cv2.circle(mascara, (i[0], i[1]), i[2], (255, 255, 255), -1)

    imgProcessada = cv2.bitwise_and(img, mascara)

    return imgProcessada

def removerPupila(img):
    """
    Remove a pupila da imagem.

    Argumentos:
        img: Imagem de entrada processada.

    Retorna:
        imgFinal: Imagem sem a pupila.
    """
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    _, threshold = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,
                               2, 500,
                               param1=1, param2=1,
                               minRadius=20, maxRadius=50)

    circles = np.uint16(np.around(circles))
    mascara = np.full_like(img, 255)

    for i in circles[0, :]:
        cv2.circle(mascara, (i[0], i[1]), i[2], (0, 0, 0), -1)

    imgFinal = cv2.bitwise_and(img, mascara)
    return imgFinal

if __name__ == '__main__':
    for imgFile in os.listdir('saída'):
        os.remove(os.path.join('saída', imgFile))

    for fileName in os.listdir('imagens'):
        img = cv2.imread(os.path.join('imagens', fileName))
        iris = segmentarIris(img)
        irisSemPupila = removerPupila(iris)

        plt.imshow(irisSemPupila)
        plt.axis('off')
        plt.savefig(os.path.join('saída', fileName), format='jpg', dpi=300)
