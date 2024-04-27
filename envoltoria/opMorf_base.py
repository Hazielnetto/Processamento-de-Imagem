
import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "caminho do arquivo"
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

_, res = cv2.threshold(topHat, 20, 255, cv2.THRESH_BINARY)
_, img_binarized = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(img_binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
maior_contorno = max(contornos, key=cv2.contourArea)

mascara = np.zeros_like(img)
cv2.drawContours(mascara, [maior_contorno], -1, 255, thickness=cv2.FILLED)
res[mascara == 0] = 0

final = cv2.medianBlur(res,3)

plt.figure(figsize=(20, 10))
plt.get_current_fig_manager().window.state('zoomed') 

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem em Escala de Cinza', fontsize=16)  
plt.grid(False)  

plt.subplot(1, 2, 2)
plt.imshow(final, cmap='gray')
plt.title('Calcificações Realçadas', fontsize=16)  
plt.grid(False)  

plt.tight_layout()

plt.show()
