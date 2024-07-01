#Alunos: Haziel Netto e Vinícius Zoz

import cv2
import os
import numpy as np
from prettytable import PrettyTable

pontos = []


def aprimoraImagems(img):
    resized = cv2.resize(img, (256, 256))
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    imgFinal = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    return adjusted


def carregaImagens(caminho):

    global imgT, alg
    caminhoEntrada = os.path.join(caminho, "entrada")
    caminhoTeste = os.path.join(caminho, "dataset")
    i = 0
    for alg in range(2):
        for imgE in os.listdir(caminhoEntrada):
            imgEntrada = cv2.imread(os.path.join(caminhoEntrada, imgE))
            aprimoraImagems(imgEntrada)
            for pasta in os.listdir(caminhoTeste):
                nomeArquivo, _ = os.path.splitext(imgE)
                if nomeArquivo == pasta:
                    pasta = os.path.join(caminhoTeste, pasta)
                    for imgT in os.listdir(pasta):
                        imgTeste = cv2.imread(
                            os.path.join(caminhoTeste, pasta, imgT))
                        if alg == 0:
                            imgSaida_sift = executaSIFT(imgEntrada, imgTeste)
                            cv2.imwrite(
                                os.path.join(caminho, 'saida',
                                             f'saida_sift_{i}.jpg'),
                                imgSaida_sift)
                        elif alg == 1:
                            imgSaida_orb = executaORB(imgEntrada, imgTeste)
                            cv2.imwrite(
                                os.path.join(caminho, 'saida',
                                             f'saida_orb_{i}.jpg'),
                                imgSaida_orb)

                    i += 1


def calculaPontos(ponto):
    geraTabela(imgT, ponto)


def geraTabela(nomeImg, resultado):
    global table
    if alg == 0:
        algoritmoNome = "SIFT"
    else:
        algoritmoNome = "ORB"
    table.add_row([nomeImg, resultado, algoritmoNome])
    table.field_names = ["Imagem", "Pontos Detectados", "Algoritmo"]


def executaSIFT(imgEntrada, imgTeste):
    sift = cv2.SIFT_create()
    return executa_algoritmo(imgEntrada, imgTeste, sift)


def executaORB(imgEntrada, imgTeste):
    orb = cv2.ORB_create()
    return executa_algoritmo(imgEntrada, imgTeste, orb)


def executa_algoritmo(imgEntrada, imgTeste, algoritmo):
    pontos_chave_entrada, descritores_entrada = algoritmo.detectAndCompute(
        imgEntrada, None)
    pontos_chave_teste, descritores_teste = algoritmo.detectAndCompute(
        imgTeste, None)

    bf = cv2.BFMatcher()
    correspondencias = bf.knnMatch(descritores_entrada, descritores_teste, k=2)

    boas_correspondencias = []
    for m, n in correspondencias:
        if m.distance < 0.75 * n.distance:
            boas_correspondencias.append(m)

    calculaPontos(len(boas_correspondencias))

    imgSaida = cv2.drawMatches(imgEntrada,
                               pontos_chave_entrada,
                               imgTeste,
                               pontos_chave_teste,
                               boas_correspondencias,
                               None,
                               flags=2)
    return imgSaida


def main():
    caminho = "E:\\Documentos\\Trabalhos FURB\\Processamento de Imagem\\HOUGH HAARCASCADE E SIFT\\placas\\amostras"
    caminhoSaida = os.path.join(caminho, 'saida')

    for imgFile in os.listdir(caminhoSaida):
        os.remove(os.path.join(caminhoSaida, imgFile))

    carregaImagens(caminho)
    print(table)


if __name__ == "__main__":
    table = PrettyTable()
    main()
