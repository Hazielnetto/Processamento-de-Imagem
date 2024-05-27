## Autores: Haziel Albuquerque Netto e Vinicius Zoz
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
from prettytable import PrettyTable

def processarImagem(imgPosKmeans):
    """
    Aplica técnicas de processamento de imagem para preparar para a extração de contornos

    Argumentos:
    imgPosKmeans: Imagem processada pelo K-means

    Retorna:
    image: Imagem após erosão e dilatação para destacar características
    """

    # Converte a imagem para escala de cinza
    imgEscalaCinza = cv2.cvtColor(imgPosKmeans, cv2.COLOR_BGR2GRAY)

    # Aplica limiarização para simplificar a imagem
    imgLimiarizada = limiarizacaoExecuta(imgEscalaCinza)

    # Cria um kernel e aplica erosão seguida por dilatação
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(imgLimiarizada, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)

    return image

def kmeansExecuta(img):
    """
    Aplica o algoritmo K-means para segmentar a imagem em cores predominantes

    Argumentos:
    img: Imagem a ser segmentada

    Retorna:
    res2: Imagem reconfigurada com cores centradas nos clusters encontrados
    """

    # Prepara a imagem para K-means
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # Define os critérios de parada do K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3

    # Aplica K-means
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def limiarizacaoExecuta(imgEscalaCinza):
    """
    Realiza limiarização automática baseada em histograma para focar em características relevantes

    Argumentos:
    imgEscalaCinza: Imagem em escala de cinza

    Retorna:
    imgLimiarizada: Imagem após aplicação de limiar
    """

    # Calcula o histograma da imagem
    histograma = cv2.calcHist([imgEscalaCinza], [0], None, [256], [0, 256])
    indHist = np.nonzero(histograma.flatten())[0]

    # Define os limites de limiarização baseados no histograma
    maisEscuro = indHist[1]
    maisClaro = indHist[2]
    minimo = maisEscuro / 2
    maximo = (maisClaro + maisEscuro) / 2

    # Aplica limiarização
    imgLimiarizada = cv2.inRange(imgEscalaCinza, minimo, maximo)
    return imgLimiarizada

def encontrarContorno(image):
    """
    Encontra contornos na imagem, útil para identificação de objetos

    Argumentos:
    image: Imagem binarizada

    Retorna:
    contorno: Lista de contornos detectados
    """

    # Extrai contornos da imagem
    contorno, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contorno

def plotarImagens(img, imgPosKmeans, imgTratada, contorno, imagemContornada):
    """
    Visualiza uma série de imagens para análise de resultados do processamento

    Argumentos:
    img: Imagem original
    imgPosKmeans: Imagem após aplicação de K-means
    imgTratada: Imagem após processamento
    contorno: Contornos encontrados na imagem
    imagemContornada: Imagem original com contornos desenhados
    """

    # Cria uma série de subplots para mostrar cada etapa do processamento
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), facecolor='gray')

    # Configura e mostra cada subplot
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
    """
    Adiciona uma entrada a uma tabela com o nome da imagem e a quantidade de contornos encontrados

    Argumentos:
    imagem: Nome da imagem processada
    contorno: Lista de contornos detectados na imagem
    """

    table.add_row([imagem, str(len(contorno))])
    table.field_names = ["Imagem", "Qtd Núcleos"]

def main():
    """
    Função principal que organiza o fluxo de processamento de várias imagens
    """

    # Define o caminho do diretório com imagens
    caminho = "datasets/"

    # Itera sobre cada imagem no diretório especificado
    for imagens in os.listdir(caminho):
        imagem = caminho + imagens
        img = cv2.imread(imagem)
        imgPosKmeans = kmeansExecuta(img)
        imgTratada = processarImagem(imgPosKmeans)
        contorno = encontrarContorno(imgTratada)
        imagemContornada = cv2.drawContours(img, contorno, -1, 255, 1)

        # Mostra as imagens processadas e atualiza a tabela
        plotarImagens(img, imgPosKmeans, imgTratada, contorno, imagemContornada)
        criarTabela(imagens, contorno)

    print(table)

# Inicializa a tabela e chama a função principal
table = PrettyTable()
main()
