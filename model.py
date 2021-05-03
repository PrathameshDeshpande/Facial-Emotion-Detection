from imports import *

# Function to create a Trained Model
def create_model():
    vggmodel = keras.applications.vgg16.VGG16()
    newmodel = Sequential()
    num = 0
    for i, layer in enumerate(vggmodel.layers):
        if i<19:
          newmodel.add(layer)
    newmodel.summary()
    # Freezing VGG network
    for layer in newmodel.layers:
      layer.trainable=False
    return newmodel

# Funtion to extract features from model
def create_features(X,model):
    vggfeatures = []
    for i, sample in enumerate(X):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 224, 224, 3))
        prediction = model.predict(sample)
        prediction = prediction.reshape((7, 7, 512))
        vggfeatures.append(prediction)
    vggfeatures = np.array(vggfeatures)
    print(vggfeatures.shape)
    return vggfeatures

