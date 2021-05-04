from imports import *
from model import create_model1,create_model2,create_features_vgg,create_features_inception


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Taking images from dataset
path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_v = train_datagen.flow_from_directory(path, target_size=(224, 224),batch_size=32,class_mode=None)
train_i = train_datagen.flow_from_directory(path, target_size=(299, 299),batch_size=32,class_mode=None)
# creating X and Y training set

X =[]
Y =[]
for img in train_v[0]:
  try:
      # converting rgb color space to lab
      lab = rgb2lab(img)
      # Appending L (Lightness in X)
      X.append(lab[:,:,0])
      # Appending color in Y
      Y.append(lab[:,:,1:] / 128)
  except:
     print('error')
X_v = np.array(X)
Y = np.array(Y)
X_v = X_v.reshape(X_v.shape+(1,))
print(X_v.shape)
print(Y.shape)

X =[]
for img in train_i[0]:
  try:
      # converting rgb color space to lab
      lab = rgb2lab(img)
      # Appending L (Lightness in X)
      X.append(lab[:,:,0])
  except:
     print('error')
X_i = np.array(X)
X_i = X_i.reshape(X_i.shape+(1,))
print(X_i.shape)


# Lets get feature vector as input for decoder
model1 = create_model1()
model2 = create_model2()
vgg_features = create_features_vgg(X_v,model1)
inception_features = create_features_inception(X_i,model2)
# We will use output of VGG model as encoder output

# Encoder with just input as we already have encoded input
encoder_input1 = Input(shape=(7, 7, 512,))
decoder_output1 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_input1)
encoder_input2 = Input(shape =(8,8,2048))
decoder_output2 = Conv2D(256, (2,2), activation='relu')(encoder_input2)
decoder_output = Add()([decoder_output1,decoder_output2])

# Decoder attached to encoder output disguised as input
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=(encoder_input1,encoder_input2), outputs=decoder_output)
model.summary()
# compiling the model
model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
filepath="/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/weights/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fitting the model
model.fit((vgg_features,inception_features), Y, validation_split = 0.2,verbose=1, epochs=10000, batch_size=32,callbacks=callbacks_list)



