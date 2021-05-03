from  imports import *
from model import create_model1,create_model2,create_features



physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)



model1 = create_model1()
model2 = create_model2()
model = tf.keras.models.load_model("/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/weights/weights-improvement-9453.hdf5")

test_path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr/im54.jpg"
test = img_to_array(load_img(test_path))
test = resize(test, (224,224), anti_aliasing=True)
test*= 1.0/255
lab = rgb2lab(test)
l = lab[:,:,0]
L = gray2rgb(l)
L = L.reshape((1,224,224,3))
#print(L.shape)
vggpred = model1.predict(L)
inception = model2.predict(L)
ab = model.predict(vggpred)
#print(ab.shape)
ab = ab*128
cur = np.zeros((224, 224, 3))
cur[:,:,0] = l
cur[:,:,1:] = ab
imsave("result.jpg", lab2rgb(cur))
