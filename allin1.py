import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np 
from keras import Model
import cv2 as cv
from keras.layers import Dense
#import matplotlib as plt
from skimage.transform import resize
#import pytorch as torch
#import tensorflow as tf
import tf2onnx
import onnx

from tensorflow.keras.preprocessing import image
model= MobileNet(weights= 'imagenet')
data= np.empty((150, 224, 224,3))
for i in range(40):
#    print('Users/avni/Desktop/mobilenet/my_data/Robocon_Logo/r{}.jpeg'.format(i+1))
    
#    assert os.path.exists("/Users/avni/Desktop/mobilenet/my_data/Robocon_Logo/r{}.jpeg".format(i+1))
    im= cv.imread('./my_data/Robocon_Logo/r{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= cv.resize(im, (224, 224))
    data[i]=im
for i in range(55):
    im= cv.imread('/Users/avni/Desktop/mobilenet/my_data/oracle_bone/o{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= resize(im, output_shape=(224, 224))
    data[i+40]=im
for i in range(55):
    im= cv.imread('/Users/avni/Desktop/mobilenet/my_data/Random_Symbols/f{}.jpeg'.format(i+1))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im , cv.COLOR_GRAY2BGR)
    im= preprocess_input(im)
    im= resize(im, output_shape=(224, 224))
    data[i+95]=im
labels= np.empty(150, dtype=int)
labels[0:40]=0
labels[40:95]=1
labels[95:]=2


training_data= np.empty((120, 224, 224, 3))
training_data[:32]= data[:32]
training_data[32:76]= data[40:84]
training_data[76:]= data[95:139]
training_labels= np.empty(120)
training_labels[:32]=0
training_labels[32:76]=1
training_labels[76:]=2
validation_data= np.empty((30, 224, 224, 3))
validation_data[:8]= data[32:40]
validation_data[8:19]= data[84:95]
validation_data[19:]= data[139:150]
validation_labels= np.empty(30)
validation_labels[:8]= 0
validation_labels[8:19]= 1
validation_labels[19:]= 2




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
    epochs=30,
    verbose=2,
    validation_data=(validation_data, validation_labels)
)

predictions= myModel.predict(training_data)
onnxm= model.save('./my_model.keras')

myModel.summary()


input_signature = [
    tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input_1')
]
onnx_model= tf2onnx.convert.from_keras(onnxm, input_signature=input_signature, opset=13)
onnx.save(onnx_model, "./model.onnx")

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
    print("prediction value is ", preds)
    decoded_preds = np.argmax(preds[0])
    if decoded_preds==0:
        print("robocon logo")
    elif decoded_preds==1:
        print("oracle bone")
    elif decoded_preds==2:
        print("random")
    else:
        print("i dont know")
    
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