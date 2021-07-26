import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization,Dense
from keras.preprocessing.image import ImageDataGenerator as Imgen
from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = pd.read_csv("/home/prathamesh/Desktop/Body_Language_Detector/data/icml_face_data.csv")
data.columns = ['emotion', 'Usage', 'pixels']
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
dataplus = pd.read_csv("/home/prathamesh/Desktop/Body_Language_Detector/data/FERPlus-master/fer2013new.csv")
x = dataplus.iloc[:,2:11].idxmax(axis=1)
e = {"anger":0,"disgust":1,"fear":2,"happiness":3,"sadness":4,"surprise":5,"neutral":6,"contempt":7,"unknown":8,"NF":9}
elist = data["emotion"].to_numpy()
emotionplus = []
for i,j in zip(x,elist):
    if(i == "contempt" or i == "unknown" or i == "NF"):
        emotionplus.append(j)
    else:
        emotionplus.append(e[i])
#print(emotionplus)
data["emotionplus"] = emotionplus


def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotionplus'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i, :, :, 0] = image / 255

    return image_array, image_label

full_train_images, full_train_labels = prepare_data(data[data['Usage']=='Training'])
test_images, test_labels = prepare_data(data[data['Usage']!='Training'])

print(full_train_images.shape)
print(full_train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images, valid_images, train_labels, valid_labels = train_test_split(full_train_images, full_train_labels, test_size=0.2, random_state=1)

print(train_images.shape)
print(valid_images.shape)
print(train_labels.shape)
print(valid_labels.shape)


def plot_all_emotions():
    N_train = train_labels.shape[0]

    sel = np.random.choice(range(N_train), replace=False, size=16)

    X_sel = train_images[sel, :, :, :]
    y_sel = train_labels[sel]

    plt.figure(figsize=[12,12])
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(X_sel[i,:,:,0], cmap='binary_r')
        plt.title(emotions[y_sel[i]])
        plt.axis('off')
    plt.show()

def plot_examples(label):
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    for i in range(5):
        idx = data[data['emotion']==label].index[i]
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        axs[i].set_title(emotions[label])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])


def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
    """ Function to plot the image and compare the prediction results with the label """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    bar_label = emotions.values()

    axs[0].imshow(test_image_array[image_number], 'gray')
    axs[0].set_title(emotions[test_image_label[image_number]])

    axs[1].bar(bar_label, pred_test_labels[image_number], color='orange', alpha=0.7)
    axs[1].grid()

    plt.show()


def vis_training(hlist, start=1):
    loss = np.concatenate([h.history['loss'] for h in hlist])
    val_loss = np.concatenate([h.history['val_loss'] for h in hlist])
    acc = np.concatenate([h.history['accuracy'] for h in hlist])
    val_acc = np.concatenate([h.history['val_accuracy'] for h in hlist])

    epoch_range = range(1, len(loss) + 1)

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range[start - 1:], loss[start - 1:], label='Training Loss')
    plt.plot(epoch_range[start - 1:], val_loss[start - 1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range[start - 1:], acc[start - 1:], label='Training Accuracy')
    plt.plot(epoch_range[start - 1:], val_acc[start - 1:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()

plot_all_emotions()
plot_examples(label=0)
plot_examples(label=1)
plot_examples(label=2)
plot_examples(label=3)
plot_examples(label=4)
plot_examples(label=5)
plot_examples(label=6)

class_weight = dict(zip(range(0, 7), (((data[data['Usage']=='Training']['emotion'].value_counts()).sort_index())/len(data[data['Usage']=='Training']['emotion'])).tolist()))
print(class_weight)

model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.summary()
h1 = model.fit(train_images, train_labels, batch_size=256, epochs=30, verbose=1, validation_data=(valid_images, valid_labels))
model.save('base_model.h5')

vis_training([h1])

#model =load_model('Dropout+normalization.h5')
test_prob = model.predict(test_images)
test_pred = np.argmax(test_prob, axis=1)
test_accuracy = np.mean(test_pred == test_labels)

print(test_accuracy)


conf_mat = confusion_matrix(test_labels, test_pred)

pd.DataFrame(conf_mat, columns=emotions, index=emotions)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=emotions,
                                figsize=(8, 8))
fig.show()