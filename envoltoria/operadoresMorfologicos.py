
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

def desfoqueMediana(img):
    """
    Aplica desfoque mediano na imagem.

    Argumentos:
    img: Imagem de entrada.

    Retorna:
    imgp: Imagem após aplicar desfoque mediano.
    """
    imgp = cv2.medianBlur(img, 3)
    return imgp

def topHatBlackHat(img):
    """
    Aplica operações Top Hat e Black Hat na imagem.

    Argumentos:
    img: Imagem de entrada.

    Retorna:
    imgTratada: Imagem após aplicar as operações.
    topHat: Imagem resultante da operação Top Hat.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    erosion = cv2.erode(img, kernel, iterations=2)
    imgTratada = (img + topHat) - erosion
    return imgTratada, topHat

def limiarizacaoMultiplos(topHat):
    """
    Aplica limiarização múltipla na imagem.

    Argumentos:
    topHat: Imagem a ser limiarizada.

    Retorna:
    resultadosTopHat: Lista contendo imagens após a limiarização.
    """
    resultadosTopHat = []
    for limiarTopHat in range(15, 255, 5):
        _, resTopHat = cv2.threshold(topHat, limiarTopHat, 255, cv2.THRESH_BINARY)
        resultadosTopHat.append(resTopHat)
    return resultadosTopHat

def combinarLimiares(resultadosTopHat, imgTratadaBinarizada):
    """
    Combina imagens binarizadas após limiarização.

    Argumentos:
    resultadosTopHat: Lista de imagens binarizadas.
    imgTratadaBinarizada: Imagem binarizada após pós-processamento.

    Retorna:
    combinado: Imagem combinada após a operação.
    """
    combinado = np.zeros_like(imgTratadaBinarizada)
    for resTopHat in resultadosTopHat:
        combinado[(resTopHat == 255) & (imgTratadaBinarizada == 255)] = 255
    return combinado

def encontrarMaiorContorno(binarizada):
    """
    Encontra o maior contorno na imagem binarizada.

    Argumentos:
    binarizada: Imagem binarizada.

    Retorna:
    maiorContorno: Maior contorno encontrado.
    """
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maiorContorno = max(contornos, key=cv2.contourArea)
    return maiorContorno

def criarMascara(img, maiorContorno):
    """
    Cria uma máscara a partir do maior contorno.

    Argumentos:
    img: Imagem original.
    maiorContorno: Maior contorno encontrado.

    Retorna:
    mascara: Máscara do maior contorno.
    """
    mascara = np.zeros_like(img)
    cv2.drawContours(mascara, [maiorContorno], -1, 255, thickness=cv2.FILLED)
    return mascara

def aplicarMascara(res, mascara):
    """
    Aplica a máscara na imagem resultante.

    Argumentos:
    res: Imagem resultante.
    mascara: Máscara a ser aplicada.

    Retorna:
    res: Imagem resultante após aplicar a máscara.
    """
    res[mascara == 0] = 0
    return res

def plotarImagens(imgOriginal, imgProcessada, imgCalcificacoes, valorOtsu):
    """
    Plota as imagens original, processada e com as calcificações detectadas.

    Argumentos:
    imgOriginal: Imagem original.
    imgProcessada: Imagem após processamento.
    imgCalcificacoes: Imagem com as calcificações detectadas.
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), facecolor='gray')

    axs[0].imshow(imgOriginal, cmap='gray')
    axs[0].set_title('Imagem Original')
    axs[0].axis('off')

    axs[1].imshow(imgProcessada, cmap='bone')
    axs[1].set_title('Imagem Processada')
    axs[1].axis('off')

    axs[2].imshow(imgCalcificacoes, cmap='bone')
    axs[2].set_title(f'Calcificações Detectadas (Limiar Otsu = {valorOtsu})')
    axs[2].axis('off')

    fig.suptitle('Análise de Imagens', fontsize=16)
    plt.subplots_adjust(wspace=0.3)

    plt.get_current_fig_manager().window.state('zoomed')

    plt.show()

def main():
    caminhoImagem = "Processamento de Imagem\\mama.png"
    ##caminhoImagem = "Processamento de Imagem\\mama2.png"
    ##caminhoImagem = "Processamento de Imagem\\mama2.jpeg"

    # Pré-processamento
    img = preprocessarImagem(caminhoImagem)
    imgTratada, topHat = topHatBlackHat(img)

    # Detecção de bordas
    otsu, imgTratadaBinarizada = cv2.threshold(topHat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu < 15:
        otsu, imgTratadaBinarizada = cv2.threshold(topHat, 15, 255, cv2.THRESH_BINARY)
    
    # Limiarização com múltiplos limiares
    resultadosLimiarizacao = limiarizacaoMultiplos(imgTratada)
    combinado = combinarLimiares(resultadosLimiarizacao, imgTratadaBinarizada)

    # Pós-processamento
    imagemFinal = desfoqueMediana(combinado)

    plotarImagens(img, imgTratada, imagemFinal, otsu)

main()
