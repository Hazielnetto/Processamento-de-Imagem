
import os
import keras.optimizers
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import utils

def treinaModelo(train_dir, validation_dir):
 # Define os parâmetros do modelo
    input_shape = (hModel, wModel, 3)
    num_classes = 4
    batch_size = 64
    epochs = 10
    kernel = (15,15)
    
    # Cria um modelo de rede neural convolucional simples    
    model = Sequential()   
     
    BatchNormalization(axis=-1, momentum=0.01)  
    utils.set_random_seed(1)
      
    model.add(Conv2D(32, kernel, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, kernel, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, kernel, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))   
    model.add(Dense(64, activation='relu'))    
    model.add(Dense(num_classes, activation='softmax'))

    # Compila o modelo
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    
    # valida a pasta que contem as imagens de treinamento e define cada subdiretório como sendo uma classe.
    # Dessa forma, todas as imagens que estão dentro da pasta rural, pertencem a classe rural.
    # É gerado um índice para cada uma das classes, índice fica de 0 a 3 nesse caso, por serem 4 classes.
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(hModel, wModel),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Repete o mesmo processo para os arquivos de validação.
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(hModel, wModel),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Obtém o índice no qual foi dividido cada classe de treinamento, conforme os subdiretórios da pasta de treinamento.
    class_indices = train_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    print(index_to_class)

    # Treina o modelo
    history = model.fit(train_generator, epochs=epochs)

    # Avalia o modelo
    accuracy = model.evaluate(validation_generator)
    print(f'Acurácia no conjunto de validação: {accuracy[1] * 100:.2f}%')

    # Exporta o modelo treinado
    model.save('modelo')

def montaImagem(classes, caminhoImagem):

    cor = None
    branco = 'white'
    preto = 'black'
    azul = '#007BFF'
    amarelo = '#FFFD55'
    vermelho = '#EB3324'
    verde = '#75F94D'

    # Defini o tamanho da imagem final baseando-se na resolução da imagem que iremos analisar
    largura_final, altura_final = Image.open(caminhoImagem).size

    # Define o tamanho dos clusters
    largura_pequena, altura_pequena = h, w

    # Cria a imagem final, definindo seu tamanho e nome
    imgFinal = Image.new("RGB", (largura_final, altura_final), "white")

    # Calcule quantas imagens pequenas cabem em largura e altura
    num_colunas = largura_final // largura_pequena
    num_linhas = altura_final // altura_pequena
    print('num_linhas', num_linhas)
    print('num_colunas', num_colunas)
        
    i=0
    for l in range(num_linhas):
        for c in range(num_colunas):

        # {0: 'Rural', 1: 'Urbano ', 2: 'Água', 3: 'Área Verde'}

            if classes[i] == 0: # Rural
                cor = amarelo
            elif classes[i] == 1: # Urbano
                cor = vermelho
            elif classes[i] == 2: # Água
                cor = azul
            elif classes[i] == 3: # Área Verde
                cor = verde

            img = ImageOps.colorize(Image.open(f'imgs/cluster(l{l})(c{c}).jpg').convert('L'), preto, branco, cor)
            pos_x = largura_pequena * c
            pos_y = altura_pequena * l
            imgFinal.paste(img, (pos_x, pos_y))
            i+=1                

    mostraImagem(imgFinal)

def mostraImagem(img): 
       
    # Mostra a imagem final
    img.show()
    #plt.axis('off')
    #plt.show()

def main(caminhoImagem):
    
        # Inicializa classes como um array
        classes = []
                
        # Abra a imagem de alta resolução
        imagem_grande = Image.open(caminhoImagem)

        # Defina o tamanho das imagens pequenas
        largura_pequena, altura_pequena = h, w

        # Obtenha as dimensões da imagem grande
        largura_grande, altura_grande = imagem_grande.size

        # Calcule quantas imagens pequenas cabem em largura e altura
        num_colunas = largura_grande // largura_pequena
        num_linhas = altura_grande // altura_pequena

        # Loop para dividir a imagem
        with tqdm(total=num_colunas * num_linhas) as barra_progresso:
            for linha in range(num_linhas):
                for coluna in range(num_colunas):
                        # Calcule as coordenadas de corte
                        left = coluna * largura_pequena
                        top = linha * altura_pequena
                        right = left + largura_pequena
                        bottom = top + altura_pequena

                        # Recorte a parte da imagem
                        imagem_pequena = imagem_grande.crop((left, top, right, bottom))

                        # Salve a imagem pequena com um nome único
                        nome_arquivo = f"./imgs/cluster(l{linha})(c{coluna}).jpg"

                        # Processa e retorna a classe do cluster
                        classes.append(processaImagem(imagem_pequena))
                        
                        # Salva o cluster como um jpg caso não exista
                        if not os.path.exists(nome_arquivo):
                            imagem_pequena.save(nome_arquivo)

                        # Atualiza a barra de progressão a cada iteração
                        barra_progresso.update(1)
                        
        # Monta a imagem final, juntando todos os clusters
        montaImagem(classes, caminhoImagem)
        
        # Feche a imagem grande
        imagem_grande.close()

def processaImagem(img):
    
    # Converte a imagem para um array de 4 elementos
    x = image.image_utils.img_to_array(np.array(img))
    x = np.expand_dims(x, axis=0)
    
    # Processa imagens no CNN
    prediction = modelo.predict(x, verbose=0)
    
    return np.argmax(prediction)

if __name__ == '__main__':
    # Utilizar o modelo exportado
    global modelo, h, w, hModel, wModel
    
    # Define o tamanho dos clusters
    tamanho = 150    
    h, w = tamanho, tamanho 
    hModel, wModel = tamanho, tamanho
    
    # Define os diretórios dos conjuntos de dados para cada classe
    train_dir = './Treinamento'
    validation_dir = './Validacao'
    imgs_teste = './imgs'

    # Recria o diretório dos clusters
    if os.path.exists(imgs_teste):        
        # Limpar pasta com comando cmdlet (python nao permite apagar pastas com conteudo)
        os.system('del /q imgs')
    elif not os.path.exists(imgs_teste):
        os.mkdir(imgs_teste)
        
    # Cria um novo modelo caso não exista  
    if not os.path.exists('modelo'):
       treinaModelo(train_dir, validation_dir)
       
    # Define o modelo a ser utilizado
    modelo = tf.keras.models.load_model('./modelo')
    
    # Inicializa classe main
    main('Imagem 2.jpg') 
    
    # Limpa a pasta imgs
    os.system('del /q imgs')   
    