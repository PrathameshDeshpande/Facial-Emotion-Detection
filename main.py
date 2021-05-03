from imports import *
from model import create_model,create_features

# Taking images from dataset
path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(224, 224),batch_size=32,class_mode=None)

# creating X and Y training set

X =[]
Y =[]
for img in train[0]:
  try:
      # converting rgb color space to lab
      lab = rgb2lab(img)
      # Appending L (Lightness in X)
      X.append(lab[:,:,0])
      # Appending color in Y
      Y.append(lab[:,:,1:] / 128)
  except:
     print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))
print(X.shape)
print(Y.shape)

# Lets get feature vector as input for decoder
newmodel = create_model()
vgg_features = create_features(X,newmodel)

# We will use output of VGG model as encoder output

# Encoder with just input as we already have encoded input
encoder_input = Input(shape=(7, 7, 512,))
# Decoder attached to encoder output disguised as input
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_input)
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
model = Model(inputs=encoder_input, outputs=decoder_output)

# compiling the model
model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
filepath="/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fitting the model
model.fit(vgg_features, Y,validation_split=0.2, verbose=1, epochs=100, batch_size=32,callbacks=callbacks_list)


