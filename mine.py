import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os 
from keras import Model
import cv2 as cv
from keras.layers import Dense
import matplotlib as plt
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
model= MobileNet(weights= 'imagenet')
data= np.empty((75, 224, 224,3))
for i in range(20):
    print('Users/avni/Desktop/mobilenet/my_data/Robocon_Logo/r{}.jpeg'.format(i+1))
    
    assert os.path.exists("/Users/avni/Desktop/mobilenet/my_data/Robocon_Logo/r{}.jpeg".format(i+1))
    im= cv.imread('/Users/avni/Desktop/mobilenet/my_data/Robocon_Logo/r{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= cv.resize(im, (224, 224))
    data[i]=im
for i in range(35):
    im= cv.imread('/Users/avni/Desktop/mobilenet/my_data/oracle_bone/o{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= resize(im, output_shape=(224, 224))
    data[i+20]=im
for i in range(20):
    im= cv.imread('/Users/avni/Desktop/mobilenet/my_data/Random_Symbols/f{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= resize(im, output_shape=(224, 224))
    data[i+55]=im
labels= np.empty(75, dtype=int)
labels[0:20]=0
labels[20:55]=1
labels[55:]=2


training_data= np.empty((48, 224, 224, 3))
training_data[:16]= data[:16]
training_data[16:32]= data[20:36]
training_data[32:]= data[55:71]
training_labels= np.empty(48)
training_labels[:16]=0
training_labels[16:32]=1
training_labels[32:]=2
validation_data= np.empty((12, 224, 224, 3))
validation_data[:4]= data[16:20]
validation_data[4:8]= data[36:40]
validation_data[8:]= data[71:75]
validation_labels= np.empty(12)
validation_labels[:4]= 0
validation_labels[4:8]= 1
validation_labels[8:]= 2




MyOutput = Dense(3, activation= 'softmax')
MyOutput= MyOutput(model.layers[-2].output)
myInput= model.input
myModel= Model(inputs= myInput, outputs= MyOutput)
for layer in myModel.layers[:-1]:
    layer.trainable= False

myModel.compile(
    loss= 'sparse_categorical_crossentropy',
    optimizer= 'adam',
    metrics= ['accuracy']
)

myModel.fit(
    x=training_data,
    y=training_labels,
    epochs=25,
    verbose=2,
    validation_data=(validation_data, validation_labels)
)

predictions= myModel.predict(training_data)

print("YAAAAAAAAAAAAAAAAAAAAAAAAAAY")


camera= cv.VideoCapture(0)
while True:
    isTrue, frame=camera.read()
    sample_image= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    sample_image= cv.cvtColor(sample_image, cv.COLOR_GRAY2BGR)
    #img = tf.keras.utils.load_img(frame, target_size=(224, 224))
    img= preprocess_input(sample_image)
    img_arr= np.empty((1, 224, 224, 3))
    img_arr[0]= resize(img, output_shape=(224, 224))
    #x = tf.keras.utils.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    preds = myModel.predict(img_arr)
    decoded_preds = np.argmax(preds[0])
    if decoded_preds==0:
        print("robocon logo")
    if decoded_preds==1:
        print("oracle bone")
    if decoded_preds==2:
        print("random")
    '''for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}. {label}: {score:.4f}")
        plt.imshow(img)
    '''
    #plt.axis('off')
    #plt.title(f"Prediction: {decoded_preds[0][1]} ({decoded_preds[0][2]*100:.2f}%)")
    #plt.show()
    cv.imshow('video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
camera.release()
cv.destroyAllWindows()