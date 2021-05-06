from V1.imports import *

# Function to create a Trained Model
def create_model1():
    vggmodel = keras.applications.vgg16.VGG16(weights="imagenet",include_top=False)
    model1 = Sequential()
    num = 0
    for i, layer in enumerate(vggmodel.layers):
        if i<19:
          model1.add(layer)
    model1.summary()
    # Freezing VGG network
    for layer in model1.layers:
      layer.trainable=False
    return model1

def create_model2():
    inv_model = ResNet50(weights="imagenet",include_top=False)
    model2 = Model(inv_model.input, inv_model.layers[-3].output)
    for layer in model2.layers:
      layer.trainable=False
    model2.summary()
    return model2

def create_model3():
    xmodel = keras.applications.Xception(weights="imagenet",include_top=False)
    model3 = Model(xmodel.input, xmodel.layers[-3].output)
    for layer in model3.layers:
        layer.trainable = False
    model3.summary()
    return model3
    model3.summary()

# Funtion to extract features from model
def create_features_vgg(X,model):
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

def create_features_resnet(X,model):
    ifeatures = []
    for i, sample in enumerate(X):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 224, 224, 3))
        prediction = model.predict(sample)
        prediction = prediction.reshape((7, 7, 2048))
        ifeatures.append(prediction)
    ifeatures = np.array(ifeatures)
    print(ifeatures.shape)
    return ifeatures

def create_features_xception(X,model):
    ifeatures = []
    for i, sample in enumerate(X):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 299, 299, 3))
        prediction = model.predict(sample)
        prediction = prediction.reshape((10, 10, 2048))
        ifeatures.append(prediction)
    ifeatures = np.array(ifeatures)
    print(ifeatures.shape)
    return ifeatures

if __name__ == '__main__':
    model1 = create_model1()
    model2 = create_model2()
    modle3 = create_model3()