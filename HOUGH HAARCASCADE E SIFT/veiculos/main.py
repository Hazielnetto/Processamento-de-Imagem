#Alunos: Haziel Netto e Vinícius Zoz

import cv2
import numpy as np 

def destacaCarros(frame):
    """
    Realça os veículos detectados no frame com retângulos.
    
    Argumentos:
    frame (array): O frame da imagem a ser processado.
    
    Returna:
    frame (array): O frame com os veículos detectados destacados.
    """
    
    carros = classifier.detectMultiScale(frame, scaleFactor=1.1,                                                        
                                                minNeighbors=2, 
                                                minSize=(40, 40))  
    for (x, y, w, h) in carros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)


    return frame

def main(caminhoVideo):
    """
    Executa o processo de detecção de veículos em um vídeo.
    
    Argumentos:
    caminhoVideo (str): O caminho do arquivo de vídeo a ser processado.
    """
    video = cv2.VideoCapture(caminhoVideo)  
    w, h, fps = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2),
            int(video.get(cv2.CAP_PROP_FPS)))
    
    fps = int(1000 / fps)     

    while video.isOpened():
        if not cv2.waitKey(fps) & 0xFF == ord('q'):      
            ret, frame = video.read()
            if not ret:
                break
            frame = destacaCarros(frame)            
            cv2.imshow('Video', cv2.resize(frame, (w, h)))
        else:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    classifier = cv2.CascadeClassifier('HOUGH HAARCASCADE E SIFT/veiculos/cars.xml')
    frameC = 0
    kernelMorph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel = (12,12)
    caminhoVideo = 'HOUGH HAARCASCADE E SIFT/veiculos/cars.mp4'
    
    main(caminhoVideo)
