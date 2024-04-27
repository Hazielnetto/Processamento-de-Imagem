
## Autores: Haziel Albuquerque Netto e Vinicius Zoz

import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocessarImagem(caminhoImagem):
        
    img = cv2.imread(caminhoImagem, cv2.IMREAD_GRAYSCALE)
    return img

def main():
    caminhoImagem = "E:\Documentos\Trabalhos FURB\Processamento de Imagem\segmentacao\datasets\linfocito00.png"

    # Pré-processamento
    img = preprocessarImagem(caminhoImagem)    

    # Detecção de bordas
    otsu, imgTratadaBinarizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(img, cmap='gray')
    plt.show()

main()
