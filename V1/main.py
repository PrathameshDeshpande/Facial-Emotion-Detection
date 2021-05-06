from imports import *
from model import create_model1, create_model2, create_features_vgg, create_features_resnet, create_features_xception, \
    create_model3
from keras.layers import concatenate

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Taking images from dataset
path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_v = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=128, class_mode=None)
train_i = train_datagen.flow_from_directory(path, target_size=(299, 299), batch_size=128, class_mode=None)
# creating X and Y training set

X = []
Y = []
for img in train_v[0]:
    try:
        # converting rgb color space to lab
        lab = rgb2lab(img)
        # Appending L (Lightness in X)
        X.append(lab[:, :, 0])
        # Appending color in Y
        Y.append(lab[:, :, 1:] / 128)
    except:
        print('error')
X_v = np.array(X)
Y = np.array(Y)
X_v = X_v.reshape(X_v.shape + (1,))
print(X_v.shape)
print(Y.shape)

x = []
for img in train_i[0]:
    try:
        # converting rgb color space to lab
        lab = rgb2lab(img)
        # Appending L (Lightness in X)
        x.append(lab[:, :, 0])
    except:
        print('error')
x = np.array(x)
x = x.reshape(x.shape + (1,))
print(x.shape)

# Lets get feature vector as input for decoder
model1 = create_model1()
model2 = create_model2()
model3 = create_model3()
vgg_features = create_features_vgg(X_v, model1)
resnet_features = create_features_resnet(X, model2)
exception_feature = create_features_xception(x, model3)
# We will use output of VGG model as encoder output


# Encoder with just input as we already have encoded input
encoder_input1 = Input(shape=(7, 7, 512,))
decoder_output1 = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_input1)
decoder_output1 = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder_output1)

encoder_input2 = Input(shape=(7, 7, 2048,))
decoder_output2 = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_input2)
decoder_output2 = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder_output2)

encoder_input3 = Input(shape=(10, 10, 2048,))
decoder_output3 = Conv2D(256, (2, 2), activation='relu')(encoder_input3)
decoder_output3 = Conv2D(256, (2, 2), activation='relu')(decoder_output3)
decoder_output3 = Conv2D(256, (2, 2), activation='relu')(decoder_output3)

decoder_output = concatenate([decoder_output1, decoder_output2, decoder_output3])

# Decoder attached to encoder output disguised as input
decoder_output = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),padding='same')(decoder_output)
decoder_output = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=(encoder_input1, encoder_input2, encoder_input3), outputs=decoder_output)
model.summary()
# compiling the model
adam = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
filepath = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/V1/weights/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fitting the model
model.fit((vgg_features, resnet_features, exception_feature), Y, validation_split=0.2, verbose=1, epochs=1000,
          batch_size=3128, callbacks=callbacks_list)
