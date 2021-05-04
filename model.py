from imports import *

# Function to create a Trained Model
def create_model1():
    vggmodel = keras.applications.vgg16.VGG16()
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
    inv_model = InceptionV3(weights="imagenet")
    model2 = Model(inv_model.input, inv_model.layers[-3].output)
    for layer in model2.layers:
      layer.trainable=False
    model2.summary()
    return model2

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

def create_features_inception(X,model):
    ifeatures = []
    for i, sample in enumerate(X):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 299, 299, 3))
        prediction = model.predict(sample)
        prediction = prediction.reshape((8, 8, 2048))
        ifeatures.append(prediction)
    ifeatures = np.array(ifeatures)
    print(ifeatures.shape)
    return ifeatures

if __name__ == '__main__':
    model1 = create_model1()
    model2 = create_model2()
