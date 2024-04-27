
## Autores: Haziel Albuquerque Netto e Vinicius Zoz

import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocessarImagem(caminhoImagem):
    """
    Carrega e converte a imagem em escala de cinza.

    Argumentos:
    caminhoImagem: Caminho da imagem a ser pré-processada.

    Retorna:
    img: Imagem pré-processada em escala de cinza.
    """
    img = cv2.imread(caminhoImagem, cv2.IMREAD_GRAYSCALE)
    return img

def main():
    caminhoImagem = "E:\Documentos\Trabalhos FURB\Processamento de Imagem\segmentacao\datasets\linfocito00.png"
    ##caminhoImagem = "Processamento de Imagem\\mama2.png"
    ##caminhoImagem = "Processamento de Imagem\\mama2.jpeg"

    # Pré-processamento
    img = preprocessarImagem(caminhoImagem)    

    # Detecção de bordas
    otsu, imgTratadaBinarizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

main()
