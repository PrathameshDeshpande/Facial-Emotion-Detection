from imports import *
from model import create_model,create_features


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

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



test_path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr/im3.jpg"
test = img_to_array(load_img(test_path))
test = resize(test, (224,224), anti_aliasing=True)
test*= 1.0/255
lab = rgb2lab(test)
l = lab[:,:,0]
L = gray2rgb(l)
L = L.reshape((1,224,224,3))
#print(L.shape)
vggpred = newmodel.predict(L)
ab = model.predict(vggpred)
#print(ab.shape)
ab = ab*128
cur = np.zeros((224, 224, 3))
cur[:,:,0] = l
cur[:,:,1:] = ab
imsave("result.jpg", lab2rgb(cur))
