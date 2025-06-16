from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

class_names = ['chicharrita', 'otras_plagas', 'planta_achicharrada', 'planta_sana']

def predict_img(img_path, model, img_size=(256, 256)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    prediction_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])
    
    return prediction_class, confidence