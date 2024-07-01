#Alunos: Haziel Netto e Vinícius Zoz

import cv2
import matplotlib.pyplot as plt
import numpy as np

def carregarImagens(diretorio, nomes):
    imagens = []
    for nome in nomes:
        imagem = cv2.imread(f"{diretorio}/{nome}")
        imagens.append(imagem)
    return imagens

def costurarImagens(img1, img2, img3):
    img1Cinza = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3Cinza = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1Cinza, None)
    kp2, des2 = sift.detectAndCompute(img2Cinza, None)
    kp3, des3 = sift.detectAndCompute(img3Cinza, None)

    bf = cv2.BFMatcher()
    matches12 = bf.knnMatch(des1, des2, k=2)
    matches23 = bf.knnMatch(des3, des2, k=2)

    goodMatches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
    goodMatches23 = [m for m, n in matches23 if m.distance < 0.75 * n.distance]

    srcPts12 = np.float32([kp1[m.queryIdx].pt for m in goodMatches12]).reshape(-1, 1, 2)
    dstPts12 = np.float32([kp2[m.trainIdx].pt for m in goodMatches12]).reshape(-1, 1, 2)
    H12, _ = cv2.findHomography(srcPts12, dstPts12, cv2.RANSAC, 5.0)

    srcPts23 = np.float32([kp3[m.queryIdx].pt for m in goodMatches23]).reshape(-1, 1, 2)
    dstPts23 = np.float32([kp2[m.trainIdx].pt for m in goodMatches23]).reshape(-1, 1, 2)
    H23, _ = cv2.findHomography(srcPts23, dstPts23, cv2.RANSAC, 5.0)

    height, width, _ = img2.shape
    img1Aligned = cv2.warpPerspective(img1, H12, (width, height))
    img3Aligned = cv2.warpPerspective(img3, H23, (width, height))

    imgFinal = cv2.addWeighted(img1Aligned, 0.5, img2, 0.5, 0)
    imgFinal = cv2.addWeighted(imgFinal, 0.67, img3Aligned, 0.33, 0)

    return imgFinal

directory = r"E:\Documentos\Trabalhos FURB\Processamento de Imagem\HOUGH HAARCASCADE E SIFT\junta imagens\imgs"
names = ["img1.png", "img2.png", "img3.png"]

images = carregarImagens(directory, names)
finalImage = costurarImagens(images[0], images[1], images[2])

plt.imshow(cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
