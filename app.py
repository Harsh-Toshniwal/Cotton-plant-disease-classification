import streamlit as st
import numpy as np
import plotly.express as px
import tensorflow as tf
from keras.utils import img_to_array
import os
import pickle
from PIL import Image
# Heading of Web App


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cnn():
    cnn = tf.keras.models.load_model('Model/cnn_model.h5')
    losses = pickle.load(open('Model/cnn_model.pkl', 'rb'))
    return cnn, losses


cnn, losses = cnn()
result1 = ["diseased_leaf", "diseased_plant", "freash_leaf", "freash_plant"]

st.markdown("<h1 style='text-align: center;'>Cotton Disease Detection</h1>",
            unsafe_allow_html=True)
nav = st.sidebar.radio("Navigation", ('Input Section', 'Result'))
img_submit = ''
if nav == 'Input Section':
    # This Input Section

    st.markdown("<p style='text-align: justify;'>This is a app where user upload the image of a cotton plant and app determine whether the plant is contagious or not</p>",
                unsafe_allow_html=True)

    # Image Upload Section
    img_upload = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    if img_upload is not None:

        # To See Detail.

        file_details = {'filename': img_upload.name,
                        'filetype': img_upload.type,
                        'filesize': img_upload.size}

        # View Uploaded Image
        st.write('\n Max Size: 200MB')
        img = Image.open(img_upload)
        pickle.dump(img, open('img.pkl', 'wb'))
        st.image(img.resize((300, 300)), caption='Image Uploaded')
        st.write('File-Size: ', file_details['filename'], '\tFile-Type: ',
                 file_details['filetype'], '\tFile-Size: ', file_details['filesize'] // 1024, 'KB')
        img = Image.open(img_upload)
        with open(os.path.join('User_Input', img_upload.name), 'wb') as f:
            f.write(img_upload.getbuffer())

    if st.button('Submit'):
        img_submit = img_upload.name
        if img_submit is not None:
            st.success(f'Image Uploaded  {img_submit}')
        else:
            st.warning('Error')


if nav == 'Result':

    # Figure of Lost Function and Accuracy function

    # optimize = {'Stochastic Gradient Descent': 'SGD', 'Adagrad': 'Adagrad',
    #             'RMSProp': 'RMSprop', 'AdaDelta': 'Adadelta', 'Adam': 'Adam'}
    # option = st.selectbox(
    #     'Optimizer', ('Stochastic Gradient Descent', 'Adagrad', 'RMSprop', 'AdaDelta', 'Adam'))

    # SGD RMSprop Adam Adadelta Adagrad Adamax Nadam Ftrl
    # num = pd.DataFrame(
    #     np.array(np.arange(1, 100).reshape(33, 3)), columns=['a', 'b', 'c'])
    with st.container():
        fig = px.line(losses, y=['accuracy', 'val_accuracy'],
                      title="Accuracy Function")
        st.plotly_chart(fig)

    with st.container():
        fig = px.line(losses, y=['loss', 'val_loss'],
                      title="Loss Function")
        st.plotly_chart(fig)
    img = pickle.load(open('img.pkl', 'rb'))
    test_image = img_to_array(img.resize((64, 64)))
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    result = result.ravel()
    result = result.tolist()
    max = result[0]
    for i in range(0, len(result)):
        # Compare elements of array with max
        if(result[i] > max):
            max = result[i]

    st.error("Largest element present in given array: " + str(round(max, 3)) +
             " And it belongs to " + str(result1[1]) + " class.")

# # Loop through the array
  # for i in range(0, len(result)):
    # Compare elements of array with max
    # if(result[i] > max):ABC
    # max = result[i]

    # print("Largest element present in given array: " + str(max) +
    # " And it belongs to " + str(result1[1]) + " class.")
