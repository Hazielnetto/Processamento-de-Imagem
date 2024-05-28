import cv2, os, time
import numpy as np
import matplotlib.pyplot as plt


def aprimoraImagems(img):

    resized = cv2.resize(img, (512, 512))
    alpha = 2.5  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    imgFinal = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    return imgFinal


def carregaImagens(caminho):

    imgEntrada = []
    imgTeste = []
    caminhoEntrada = caminho + r"\entrada"
    caminhoTeste = caminho + r"\dataset"
    i = 0

    for imgT in os.listdir(caminhoTeste):
        imgTeste = cv2.imread(os.path.join(caminhoTeste, imgT),
                              cv2.IMREAD_GRAYSCALE)
        for imgE in os.listdir(caminhoEntrada):
            imgEntrada = cv2.imread(os.path.join(caminhoEntrada, imgE))
            imgEntrada = aprimoraImagems(imgEntrada)
            imgSaida = executaSIFT(imgEntrada, imgTeste)

            cv2.imwrite(os.path.join(caminho, 'saida', f'saida{[i]}.jpg'),
                        imgSaida)
            i += 1

    return imgSaida


def executaSIFT(imgEntrada, imgTeste):

    # Inicializar o detector SIFT
    sift = cv2.SIFT_create()

    # Encontrar os pontos-chave e descritores nas imagens
    pontos_chave_entrada, descritores_entrada = sift.detectAndCompute(
        imgEntrada, None)
    pontos_chave_teste, descritores_teste = sift.detectAndCompute(
        imgTeste, None)

    # Realizar a correspondência entre os descritores
    bf = cv2.BFMatcher()
    correspondencias = bf.knnMatch(descritores_entrada, descritores_teste, k=2)

    # Filtrar correspondências com base na distância
    boas_correspondencias = []
    for m, n in correspondencias:
        if (m.distance > 0.25 * n.distance) and (m.distance
                                                 < 0.8 * n.distance):
            boas_correspondencias.append(m)

    # Desenhar as correspondências na imagem de saída
    imgSaida = cv2.drawMatches(imgEntrada,
                               pontos_chave_entrada,
                               imgTeste,
                               pontos_chave_teste,
                               boas_correspondencias,
                               None,
                               flags=2)

    # Mostrar a imagem de saída
    cv2.imshow("Correspondências e Retângulos", imgSaida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imgSaida

def main():
    caminho = "E:\\Documentos\\Trabalhos FURB\\Processamento de Imagem\\HOUGH HAARCASCADE E SIFT\\placas\\amostras"
    caminhoSaida = os.path.join(caminho, 'saida')

    for imgFile in os.listdir(caminhoSaida):
        os.remove(os.path.join(caminhoSaida, imgFile))

    carregaImagens(caminho)


if __name__ == "__main__":
    main()
