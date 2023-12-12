from django.shortcuts import render
import tensorflow as tf
import numpy as np
import scipy
import cv2
from PIL import Image
# Create your views here.

model = tf.keras.models.load_model("webapi/mobilenet.keras")
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def clahe_img(img):
    # Convert the image to 8-bit unsigned integer format if not already in that format
    img = cv2.convertScaleAbs(img)

    # Split the image into its color channels
    r, g, b = cv2.split(img)

    # Apply CLAHE to each channel separately
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    op_R = clahe.apply(r)
    op_G = clahe.apply(g)
    op_B = clahe.apply(b)

    # Merge the CLAHE-enhanced channels back together
    img_op_clahe = cv2.merge((op_R, op_G, op_B))

    return img_op_clahe

def index(request):
    output_class = None
    if request.method == 'POST':
        plant_image = request.FILES.get('plant_image')
        if plant_image:  # Check if the file exists
            nparr = np.fromstring(plant_image.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV format
        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            image = image / 255.0

            # image = clahe_img(image)  # Apply your CLAHE function
        
            image = np.expand_dims(image, axis=0)

            pred = model.predict(image)
            output_class = class_names[np.argmax(pred)]
    print(output_class)
    context = {
            "output_class" : output_class
        }
    return render(request, "index.html", context)




    
