# Autores: Haziel Netto e Vinicius Zoz

import cv2
import numpy as np

# Carrega o classificador Haar Cascade para detecção de veículos
classifier = cv2.CascadeClassifier('HOUGH HAARCASCADE E SIFT/veiculos/cars.xml')
historicoDetect = []
frameCount = 0 

def destacaCarros(frame, windowSize=2, drawInterval=2):
    """
    Realça os veículos detectados no frame com retângulos.
    
    Argumentos:
    frame (array): O frame da imagem a ser processado.
    windowSize (int): O número de frames anteriores a considerar para suavização.
    drawInterval (int): O intervalo de frames para desenhar os retângulos.
    
    Returna:
    frame (array): O frame com os veículos detectados destacados.
    """
    global historicoDetect    
    global frameCount 

    imgGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if frameCount % drawInterval == 0:
        carros = classifier.detectMultiScale(imgGray, scaleFactor=1.1, 
                                                        minNeighbors=3, 
                                                        minSize=(20, 20))        
        historicoDetect.append(carros)
    
    if len(historicoDetect) > windowSize:
        historicoDetect = historicoDetect[-windowSize:]
    
    carrosSuavizados = [tuple(map(int, p)) for historico in historicoDetect 
                                            for p in historico]
    
    for (x, y, w, h) in carrosSuavizados:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

    frameCount += 1
    return frame

def main(caminhoVideo):
    """
    Executa o processo de detecção de veículos em um vídeo.
    
    Argumentos:
    caminhoVideo (str): O caminho do arquivo de vídeo a ser processado.
    """
    video = cv2.VideoCapture(caminhoVideo)

    while video.isOpened():        
        ret, frame = video.read()
        if not ret:
            break

        frame = destacaCarros(frame)
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(30) == ord('q'): # Tecla para interromper a execução
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    caminhoVideo = 'HOUGH HAARCASCADE E SIFT/veiculos/cars.avi'
    main(caminhoVideo)
