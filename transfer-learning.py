Training with a Larger Dataset - Cats and Dogs
In the previous lab you trained a classifier with a horses-v-humans dataset. You saw that despite getting great training results, when you tried to do classification with real images, there were many errors, due primarily to overfitting -- where the network does very well with data that it has previously seen, but poorly with data it hasn't!

In this lab you'll look at a real, and very large dataset, and see the impact this has to avoid overfitting.

Trainamento com Dataset do Kagle Cats and Dogs usando Colab para processamento e python e algumas bibliotecas como keras, panda , tensorflow...

Projeto original de github.com/lmoroney - traduzido, melhorado e atualizado por github.com/pnakamoto
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#IMPORTAR AS BIBLIOTECAS

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BAIXAR O DATASET (CHECAR NO SITE DA MICROSOFT PARA FUTURAS ATUALIZAÇÕES DE URL/DIRETORIO)
# If the URL doesn't work, visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the dataset
# Note: This is a very large dataset and will take time to download

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

#RESULTADO ESPERADO
--2024-12-15 00:04:40--  https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
Resolving download.microsoft.com (download.microsoft.com)... 23.197.2.18, 2600:1408:ec00:889::317f, 2600:1408:ec00:887::317f
Connecting to download.microsoft.com (download.microsoft.com)|23.197.2.18|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 824887076 (787M) [application/octet-stream]
Saving to: ‘/tmp/cats-and-dogs.zip’

/tmp/cats-and-dogs. 100%[===================>] 786.67M   118MB/s    in 7.3s    

2024-12-15 00:04:47 (108 MB/s) - ‘/tmp/cats-and-dogs.zip’ saved [824887076/824887076]
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#LISTAR QUANTOS ARQUIVOS TEM NO DIRETORIO PARA VERIFICAR SE O DOWNLOAD OCORREU CONFORME O ESPERADO

print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

# Expected Output:
# 12501
# 12501

#SAIDA:
12501
12501
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CRIAR E VERIFICAR SE OS DIRETORIOS FORAM CRIADOS CORRETAMENTE 
try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Este código é responsável por dividir um conjunto de dados de imagens em dois subconjuntos: treinamento e teste, garantindo uma proporção definida por split_size. Ele também ignora arquivos corrompidos ou com tamanho zero.

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring

#SAIDA
666.jpg is zero length, so ignoring.
11702.jpg is zero length, so ignoring.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Lista diretorio

print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

# Expected output:
# 11250
# 11250
# 1250
# 1250

#SAIDA
11250
11250
1250
1250
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Este código define um modelo de rede neural convolucional (CNN) para classificar imagens em duas categorias (binário), como "gatos" e "cães".

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, models

# Definição do modelo usando Input para evitar o aviso
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),  # Definição explícita da entrada
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Substitua 'lr' por 'learning_rate'
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

# Resumo do modelo
model.summary()
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TensorFlow: Usado para criar e treinar o modelo de deep learning.
#RMSprop: Um otimizador que ajusta os pesos da rede neural durante o treinamento, baseado no gradiente do erro.

from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

# Definição do modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Para classificação binária
])

# Compilação do modelo
model.compile(
    optimizer=RMSprop(learning_rate=0.001),  # Substituição de lr por learning_rate
    loss='binary_crossentropy',  # Perda para classificação binária
    metrics=['accuracy']  # Métrica de desempenho
)

# Exibir o resumo do modelo (opcional, mas útil para verificar a estrutura)
model.summary()

#SAIDA
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_3 (Conv2D)                    │ (None, 148, 148, 16)        │             448 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 74, 74, 16)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 72, 72, 32)          │           4,640 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_4 (MaxPooling2D)       │ (None, 36, 36, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 34, 34, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_5 (MaxPooling2D)       │ (None, 17, 17, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_1 (Flatten)                  │ (None, 18496)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 512)                 │       9,470,464 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │             513 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 9,494,561 (36.22 MB)
 Trainable params: 9,494,561 (36.22 MB)
 Non-trainable params: 0 (0.00 B)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Ele transforma imagens brutas armazenadas em diretórios (TRAINING_DIR e VALIDATION_DIR) em batches de dados que podem ser usados pelo modelo.
#Aplica o pré-processamento básico nas imagens, como normalização, redimensionamento e definição do formato das classes.
#Classe ImageDataGenerator do TensorFlow/Keras para carregar as imagens diretamente de diretórios, além de pré-processá-las. Aqui está uma explicação detalhada:

TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=250,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=250,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.

#SAIDA 
Found 22498 images belonging to 2 classes.
Found 2500 images belonging to 2 classes.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Note that this may take some time.
# O PROJETO INICIAL TINHA 10 EPOCAS POREM QUANDO RODAVA O CODIGO SEMPRE IDENTIFICA COMO GATO QUALQUER IMAGEM, SOLUCAO QUE ENCONTREI FOI DEFINIR ALTERAR QUANTIDADES de EPOCAS para MAIOR ACCURACY e MENOR LOSS(Hiperparamêtros)

%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
# ADICIONADO HISTORY NO MESMO BLOCO DE CODIGO PARA EVITAR CONFLITOS
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

acc = history.history['accuracy']  # Certifique-se que o 'history' existe e foi treinado
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.figure()  # Criar uma nova figura
plt.plot(epochs, acc, 'r', label="Training Accuracy")  # Adicionar legenda
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()  # Mostrar as legendas
plt.grid()    # Adicionar uma grade para melhor visualização

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure()  # Criar uma nova figura
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Mostrar os gráficos
plt.show()

print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])
print("Training Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])

#SAIDA / GRAFICO do RESULTADO DE ACURACIA E PERDA VIDE EM ARQUIVOS DESTE REPOSITORIO 

Epoch 1/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 617s 7s/step - accuracy: 0.8858 - loss: 0.2722 - val_accuracy: 0.8164 - val_loss: 0.4229
Epoch 2/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 600s 7s/step - accuracy: 0.9055 - loss: 0.2342 - val_accuracy: 0.8148 - val_loss: 0.4501
Epoch 3/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 610s 6s/step - accuracy: 0.9247 - loss: 0.1945 - val_accuracy: 0.8216 - val_loss: 0.4112
Epoch 4/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 584s 6s/step - accuracy: 0.9462 - loss: 0.1465 - val_accuracy: 0.8228 - val_loss: 0.4861
Epoch 5/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 625s 6s/step - accuracy: 0.9552 - loss: 0.1203 - val_accuracy: 0.8284 - val_loss: 0.5120
Epoch 6/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 589s 6s/step - accuracy: 0.9620 - loss: 0.1156 - val_accuracy: 0.8080 - val_loss: 0.6378
Epoch 7/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 619s 6s/step - accuracy: 0.9712 - loss: 0.0892 - val_accuracy: 0.8248 - val_loss: 0.6169
Epoch 8/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 607s 7s/step - accuracy: 0.9789 - loss: 0.0695 - val_accuracy: 0.8216 - val_loss: 0.4674
Epoch 9/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 665s 7s/step - accuracy: 0.9892 - loss: 0.0474 - val_accuracy: 0.8176 - val_loss: 0.6779
Epoch 10/10
90/90 ━━━━━━━━━━━━━━━━━━━━ 602s 7s/step - accuracy: 0.9848 - loss: 0.0619 - val_accuracy: 0.8172 - val_loss: 0.7102

Training Accuracy: [0.8858120441436768, 0.9029691815376282, 0.9135034084320068, 0.937416672706604, 0.9503511190414429, 0.9601297974586487, 0.964130163192749, 0.9644857048988342, 0.9720864295959473, 0.9761756658554077]
Validation Accuracy: [0.8163999915122986, 0.8148000240325928, 0.8216000199317932, 0.8227999806404114, 0.8284000158309937, 0.8080000281333923, 0.8248000144958496, 0.8216000199317932, 0.8176000118255615, 0.8172000050544739]
Training Loss: [0.2674766480922699, 0.23620915412902832, 0.21885797381401062, 0.16436755657196045, 0.1306021511554718, 0.12620949745178223, 0.10884514451026917, 0.12154799699783325, 0.08938798308372498, 0.10195577144622803]
Validation Loss: [0.422863632440567, 0.4501042366027832, 0.4112311005592346, 0.4860658347606659, 0.5120224356651306, 0.6378393173217773, 0.6169219613075256, 0.46742451190948486, 0.677918553352356, 0.7102173566818237]
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Here's a codeblock just for fun. You should be able to upload an image here 
# and have it classified without crashing
# PARA TESTAR O APRENDIZADO DE MAQUINA E SUBIR OS FOTOS PARA CLASSIFICAR

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a dog")
  else:
    print(fn + " is a cat")

#SAIDA 
2 ficheiros
images (1).jpeg(image/jpeg) - 28249 bytes, last modified: 09/12/2024 - 100% done
American_Bully_Stud_Male_(11527292106).jpg(image/jpeg) - 580235 bytes, last modified: 09/12/2024 - 100% done
Saving images (1).jpeg to images (1) (1).jpeg
Saving American_Bully_Stud_Male_(11527292106).jpg to American_Bully_Stud_Male_(11527292106) (1).jpg
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
[0.]
images (1) (1).jpeg is a cat
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
[1.]
American_Bully_Stud_Male_(11527292106) (1).jpg is a dog
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
