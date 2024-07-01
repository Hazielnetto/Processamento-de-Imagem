
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

modelo = tf.keras.models.load_model('./modelo')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        './Validacao',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

accuracy = modelo.evaluate(validation_generator)    
print(f'Acurácia no conjunto de validação: {accuracy[1] * 100:.2f}%')
