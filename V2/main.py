from V1.imports import *
from V1.model import create_model1, create_model2, create_features_vgg, create_features_resnet, create_features_xception, create_model3
from keras.layers import concatenate

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Taking images from dataset
path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=128, class_mode=None)

X = []
Y = []
for img in train[0]:
    try:
        # converting rgb color space to lab
        lab = rgb2lab(img)
        # Appending L (Lightness in X)
        X.append(lab[:, :, 0])
        # Appending color in Y
        Y.append(lab[:, :, 1:] / 128)
    except:
        print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape + (1,))
print(X.shape)
print(Y.shape)

model1 = create_model1()
vgg_features = create_features_vgg(X, model1)

#Encoder
encoder_input = Input(shape=(7, 7, 512,))
#Decoder
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

model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
adam = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
filepath = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/V2/weights/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(vgg_features, Y, verbose=1, validation_split=0.1,epochs=100, batch_size=128, callbacks=callbacks_list)

testpath = '/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr/im22.jpg'

test = img_to_array(load_img(testpath))
test = resize(test, (224,224), anti_aliasing=True)
test*= 1.0/255
lab = rgb2lab(test)
l = lab[:,:,0]
L = gray2rgb(l)
L = L.reshape((1,224,224,3))
#print(L.shape)
vggpred = model1.predict(L)
ab = model.predict(vggpred)
#print(ab.shape)
ab = ab*128
cur = np.zeros((224, 224, 3))
cur[:,:,0] = l
cur[:,:,1:] = ab
imsave("result2.jpg", lab2rgb(cur))