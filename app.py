import streamlit as st
from  V1.imports import *
from V1.model import create_model1,create_model2,create_model3
import cv2


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

st.markdown("<h1 style='text-align: center; color: black;'>📷 Auto-Colorization 📷</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Made By PDP with ❤</h1>", unsafe_allow_html=True)

st.title("Welcome to the Project 🧑🏽‍💻")
st.write("⭐ A project which uses Auto Encoders to Convert Grayscale Images to Colored Images⭐")
uploaded_file = st.file_uploader("🌌 🌆Choose an image...",type=['jpg','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption='👉Uploaded Image.👈', use_column_width=True)
    st.write("")

    name = ".".join(uploaded_file.name.split("/"))
    image_path = os.path.join("/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/mirflickr25k/mirflickr", name)

    model1 = create_model1()
    model2 = create_model2()
    model3 = create_model3()

    model = tf.keras.models.load_model("/home/prathamesh/Desktop/ML_Projects/Auto_Colorization/V1/weights/end_model_V2")
    image = cv2.imread(image_path)
    test1 = img_to_array(load_img(image_path))
    shape = test1.shape
    test1 = resize(test1, (224, 224), anti_aliasing=True)
    test1 *= 1.0 / 255
    lab = rgb2lab(test1)
    l1 = lab[:, :, 0]
    L1 = gray2rgb(l1)
    L1 = L1.reshape((1, 224, 224, 3))
    # print(L.shape)
    vggpred = model1.predict(L1)
    resnet = model2.predict(L1)

    testx = img_to_array(load_img(image_path))
    testx = resize(testx, (299, 299), anti_aliasing=True)
    testx *= 1.0 / 255
    lab2 = rgb2lab(testx)
    l2 = lab2[:, :, 0]
    L2 = gray2rgb(l2)
    L2 = L2.reshape((1, 299, 299, 3))
    # print(L.shape)
    exception = model3.predict(L2)

    ab = model.predict((vggpred, resnet, exception))
    # print(ab.shape)
    ab = ab * 128
    cur = np.zeros((224, 224, 3))
    cur[:, :, 0] = l1
    cur[:, :, 1:] = ab
    cur = lab2rgb(cur)
    cur = cv2.resize(cur, (448, 448))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    st.image(gray, caption='👉🌌Grayscale Image🌌👈', use_column_width=True)
    st.image(cur, caption='👉🌆Color Converted Image🌆👈', use_column_width=True)
    st.image(image, caption='👉🌇Ground Truth🌇👈', use_column_width=True)
