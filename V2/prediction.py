from V1.imports import *
from V1.model import create_model1, create_model2, create_model3
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model1 = create_model1()

model = tf.keras.models.load_model(
    "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/V2/weights/weights-improvement-100.hdf5")

test_path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr/im114.jpg"
mage = cv2.imread(test_path)
test = img_to_array(load_img(test_path))
test = resize(test, (224, 224), anti_aliasing=True)
test *= 1.0 / 255
lab = rgb2lab(test)
l = lab[:, :, 0]
L = gray2rgb(l)
L = L.reshape((1, 224, 224, 3))
# print(L.shape)
vggpred = model1.predict(L)

ab = model.predict(vggpred)
# print(ab.shape)
ab = ab * 128
cur = np.zeros((224, 224, 3))
cur[:, :, 0] = l
cur[:, :, 1:] = ab
imsave("result.jpg", lab2rgb(cur))
cv2.imshow("window", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
