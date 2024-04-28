
## Autores: Haziel Albuquerque Netto e Vinicius Zoz

import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def processarImagem(imgPosKmeans):
    imgEscalaCinza = cv2.cvtColor(imgPosKmeans, cv2.COLOR_BGR2GRAY)
    imgLimiarizada = fazLimiarizacao(imgEscalaCinza)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(imgLimiarizada, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    
    return image

def fazKmeans(image):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

def fazLimiarizacao(imgEscalaCinza):    
    histograma = cv2.calcHist([imgEscalaCinza], [0], None, [256], [0, 256])    
    indHist = np.nonzero(histograma.flatten())[0] 

    maisEscuro = indHist[1]   
    maisClaro = indHist[2]  

    minimo = maisEscuro / 2  
    maximo = (maisClaro + maisEscuro) / 2

    imgLimiarizada = cv2.inRange(imgEscalaCinza, minimo, maximo)

    return imgLimiarizada

def encontrarContorno(image):
    """
    Encontra o maior contorno na imagem binarizada.

    Argumentos:
    binarizada: Imagem binarizada.

    Retorna:
    maiorContorno: Maior contorno encontrado.
    """    
    contorno, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contorno

def plotarImagens(img, imgPosKmeans, imgTratada, contorno, imagemContornada):
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), facecolor='gray')
            
    axs[0].imshow(img)
    axs[0].set_title('Imagem Original')
    axs[0].axis('off')
    
    axs[1].imshow(imgPosKmeans)
    axs[1].set_title('Imagem Processada')
    axs[1].axis('off')
             
    axs[2].imshow(imgTratada, cmap = "gray")
    axs[2].set_title(f'Núcleo Destacado')
    axs[2].axis('off')
    
    axs[3].imshow(imagemContornada)
    axs[3].set_title(f'Núcleos Detectados: {str(len(contorno))}')
    axs[3].axis('off')

    fig.suptitle('Segmentação de Objetos', fontsize=16)
    plt.subplots_adjust(wspace=0.3)

    plt.show()

def criarTabela(imagem, contorno):    
    table.add_row([imagem, str(len(contorno))])    
    table.field_names = ["Imagem", "Qtd Núcleos"]

    print(table)    

def main():
    caminho = "segmentacao/datasets/"
    for imagens in os.listdir(caminho):               

        imagem = caminho + imagens
        img = cv2.imread(imagem)
        imgPosKmeans = fazKmeans(img)
        imgTratada = processarImagem(imgPosKmeans)
        contorno = encontrarContorno(imgTratada)
        imagemContornada = cv2.drawContours(img, contorno, -1, 255, 1)

        plotarImagens(img, imgPosKmeans, imgTratada, contorno, imagemContornada)
        
        criarTabela(imagens, contorno)
        
table = PrettyTable()

main()
