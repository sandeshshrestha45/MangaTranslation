from keras.models import load_model
import keras
import tensorflow
import numpy as np
import cv2 
import tensorflow.keras.backend as K



with open('model/gender/config.json', 'r') as f:
    model_config = f.read()

model = tensorflow.keras.models.model_from_json(model_config) 
model.load_weights ("model/gender/model.weights.h5")
# print(model.summary())

def get_gender(img):
    # img = cv2.imread("mangas/gender/male/-58-_jpg.rf.e8fd713986159a02d4f0dcb786947a08-9.jpg")
    img = cv2.resize(img,(120,120))
    expand = np.expand_dims(img,axis=0)
    predictions = model.predict(expand)
    if K.sigmoid(predictions[0][0]) < 0.5:
        return "Female"
    if K.sigmoid(predictions[0][0]) >=0.5:
        return "Male"
    print(K.sigmoid(predictions[0][0]))


if __name__ == "__main__":

    img = cv2.imread("img/translated_img/large17.jpg")
    print(get_gender(img))