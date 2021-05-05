from  imports import *
from model import create_model1,create_model2
import cv2




def prediction():
   physical_devices = tf.config.experimental.list_physical_devices('GPU')
   if len(physical_devices) > 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)

   model1 = create_model1()
   model = tf.keras.models.load_model("/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/weights/weights-improvement-64.hdf5")

   test_path = "/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr/im100.jpg"
   image = cv2.imread(test_path)
   test1 = img_to_array(load_img(test_path))
   test1= resize(test1, (224,224), anti_aliasing=True)
   test1*= 1.0/255
   lab = rgb2lab(test1)
   l1 = lab[:,:,0]
   L1 = gray2rgb(l1)
   L1 = L1.reshape((1,224,224,3))
   #print(L.shape)
   vggpred = model1.predict(L1)

   ab = model.predict(vggpred)
   #print(ab.shape)
   ab = ab*128
   cur = np.zeros((224, 224, 3))
   cur[:,:,0] = l1
   cur[:,:,1:] = ab
   imsave("result.jpg", lab2rgb(cur))
   cv2.imshow("window",image)
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   cv2.imshow("gray",gray)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

if __name__ == '__main__':
   prediction()