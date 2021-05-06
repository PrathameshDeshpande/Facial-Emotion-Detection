from  imports import *
from model import create_model1,create_model2,create_model3
import cv2



#V2 OP
def prediction():
   physical_devices = tf.config.experimental.list_physical_devices('GPU')
   if len(physical_devices) > 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)

   model1 = create_model1()
   model2 = create_model2()
   model3 = create_model3()
   model = tf.keras.models.load_model("/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/V1/weights/end_model_V2")

   test_path = "/home/prathamesh/Downloads/sunflower.jpg"
   image = cv2.imread(test_path)
   test1 = img_to_array(load_img(test_path))
   shape = test1.shape
   print(shape)
   test1= resize(test1, (224,224), anti_aliasing=True)
   test1*= 1.0/255
   lab = rgb2lab(test1)
   l1 = lab[:,:,0]
   L1 = gray2rgb(l1)
   L1 = L1.reshape((1,224,224,3))
   #print(L.shape)
   vggpred = model1.predict(L1)
   resnet = model2.predict(L1)

   testx = img_to_array(load_img(test_path))
   testx= resize(testx, (299,299), anti_aliasing=True)
   testx*= 1.0/255
   lab2 = rgb2lab(testx)
   l2 = lab2[:,:,0]
   L2 = gray2rgb(l2)
   L2 = L2.reshape((1,299,299,3))
   #print(L.shape)
   exception = model3.predict(L2)

   ab = model.predict((vggpred,resnet,exception))
   #print(ab.shape)
   ab = ab*128
   cur = np.zeros((224, 224, 3))
   cur[:,:,0] = l1
   cur[:,:,1:] = ab
   cur = lab2rgb(cur)
   cur = cv2.resize(cur, (shape[0],shape[1]))
   imsave("result.jpg", cur)
   cv2.imshow("window",image)
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   cv2.imshow("gray",gray)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

if __name__ == '__main__':
   prediction()