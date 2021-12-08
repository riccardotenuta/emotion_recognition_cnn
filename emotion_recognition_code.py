import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.s. pd.read_csv)

from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os

from google.colab import drive
drive.mount('/content/drive')

# legge il dataset (file CSV) da Google Drive
data = pd.read_csv('/content/drive/MyDrive/fer2013.csv')

# spiegazione ed esempi sulla composizione del dataset

data.Usage.value_counts()

emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
emotion_counts

# plot del dataset suddiviso in base alle classi

plt.figure(figsize=(6,4))
sns.barplot(emotion_counts.emotion, emotion_counts.number)
plt.title('Class distribution')
plt.ylabel('Number', fontsize=12)
plt.xlabel('Emotions', fontsize=12)
plt.show()

# stampa di esempi di immagini del dataset (48 x 48) con classificazione

def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotion])

plt.figure(0, figsize=(16,10))
for i in range(1,8):
    face = data[data['emotion'] == i-1].iloc[0]
    img = row2image(face)
    plt.subplot(2,4,i)
    plt.imshow(img[0])
    plt.title(img[1])

plt.show() 
    
# suddivisione dei dati del dataset in base al loro utilizzo

data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()
print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(data_train.shape, data_val.shape, data_test.shape))

print(data_train)

# barplot class distribution of train, val and test
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def setup_axe(axe,df,title):
    df['emotion'].value_counts(sort=False).plot(ax=axe, kind='bar', rot=0)
    axe.set_xticklabels(emotion_labels)
    axe.set_xlabel("Emotions")
    axe.set_ylabel("Number")
    axe.set_title(title)
    
    # set individual bar lables using above list
    for i in axe.patches:
        # get_x pulls left or right; get_height pushes up or down
        axe.text(i.get_x()-.05, i.get_height()+120, \
                str(round((i.get_height()), 2)), fontsize=14, color='dimgrey',
                    rotation=0)

   
fig, axes = plt.subplots(1,3, figsize=(20,8), sharey=True)
setup_axe(axes[0],data_train,'train')
setup_axe(axes[1],data_val,'validation')
setup_axe(axes[2],data_test,'test')
plt.show()

width, height = 48, 48
num_classes = 7

# verifico che gli array di pixel siano delle stringhe, prima di convertirli in interi

for pixels_sequence in data_train['pixels']:
  print(type(pixels_sequence[0]))

# conversione e normalizzazione dei dati del dataset 
# per essere dati in input alla rete neurale

def CRNO(df, dataName):
    # coverte gli array di pixel (stringhe) in array di interi
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])

    # reshape e normalizzazione delle immagini in bianco e nero (divisione per 255.0)
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1,width, height,1)/255.0  

    # one-hot encoding label, e.s. class 3 to [0,0,0,1,0,0,0]
    # ogni categoria/classe viene rappresentata da un vettore binario
    data_Y = to_categorical(df['emotion'], num_classes)  
    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

    
train_X, train_Y = CRNO(data_train, "train") #training data
val_X, val_Y     = CRNO(data_val, "val") #validation data
test_X, test_Y   = CRNO(data_test, "test") #test data

print(train_X)
print(train_Y)

# creazione del modello

model = Sequential()
#Block-1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#Block-6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#Block-7
model.add(Dense(7,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# kernel_initializer='he_normal' inizializza i pesi del layer

# loss='categorical_crossentropy'

# Calcola la loss tra le etichette e le previsioni.
# Si utilizza questa funzione di loss incrociata quando 
# sono presenti due o più classi di etichette. Ci aspettiamo che le 
# etichette vengano fornite in una rappresentazione one-hot encoding label

# optimizer=Adam(learning_rate=0.001)

# Adam è uno degli algortimi di ottimizzazione più diffusi e veloci per
# il training, il valore di learning rate indica quanto velocemente la 
# loss converge a 0

# The learning rate is a tuning parameter in an optimization algorithm that 
# determines the step size at each iteration while moving toward a minimum of a loss function

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

epo = 25
model.fit(train_X,train_Y,batch_size=32,epochs=epo,validation_data=(val_X, val_Y))

print(model.summary())
print(model.history.history.keys())

# plot dell'accuracy e val_accuracy in funzione delle epoche

plt.figure(figsize=(10,5))
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plot della loss e della val_loss in funzione delle epoche 

plt.figure(figsize=(10,5))
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# calcolo della loss e dell'accuracy sui dati di test

loss_and_metrics = model.evaluate(test_X,test_Y)
print(loss_and_metrics)

# salvataggio della struttura del modello e dei suoi pesi in locale

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write((model_json))
  model.save_weights("model.h5")
print("Saved model to disk")

# funzione JavaScript che permette di scattare la foto e salvarla in locale

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
  
from google.colab.patches import cv2_imshow
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import imutils
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# import del modello e dei suoi pesi precedentemente salvato

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

print(model.summary())

# richiamo funzione per scattare la foto e la memorizzo nel file photo.jpg
image_file = take_photo()

# oggetto della libreria OpenCv che serve a riconoscere i volti in un'immagine
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'/haarcascade_frontalface_default.xml')

emotion_detection = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# leggo l'immagine appena scattata
image = cv2.imread(image_file)

# converto l'immagine in bianco e nero
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detection dei volti presenti nella foto
faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)

for (x,y,w,h) in faces:

    # disegno sull'immagine un rettangolo attorno ad ogni volto
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    
    # ritaglio la sezione del volto 
    roi_gray = gray_image[y:y+h,x:x+w]
    cv2_imshow(roi_gray)

    # reshape del volto (48 x 48 pixel)
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    cv2_imshow(roi_gray)

    # l'if ritorna true se la somma totale dei valore nell'array roi_gray è diversa da 0, che significa che almeno un volto è stato riconosciuto
    if np.sum([roi_gray]) != 0:

            # conversione di roy_gray in array di float e normalizzazione dei pixel (divisione per 255.0)
            roi = roi_gray.astype('float')/255.0

            # Converts a PIL Image instance to a Numpy array
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = model.predict(roi)[0]
            print(preds)
            label = emotion_detection[preds.argmax()]
            label_position=(x,y)

            # scrivo sull'immagine l'emozione che ha la percentuale più alta nelle predizioni
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
            cv2.putText(image,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
  
cv2_imshow(image)

cv2.destroyAllWindows()
