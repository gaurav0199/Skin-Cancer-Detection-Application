import os
import numpy as np
import pickle
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

model = load_model('models/best_model.h5')
with open('models/class_names.pkl', 'rb') as f:
    classes = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def welcome():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction='No file uploaded',image_url=None)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', prediction='No selected file',image_url=None)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        processed_image = preprocess_image(image, target_size=(180, 180))
        prediction = model.predict(processed_image).argmax(axis=1)[0]
        predicted_class = classes[prediction]

        image_url = url_for('static', filename='uploads/' + filename)

        if prediction == 0:
            info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. AK is a type of precancer, which means that if you don't treat the condition, it could turn into cancer. Without treatment, AK can lead to a type of skin cancer called squamous cell carcinoma."
        
        elif prediction == 1:
            info = "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off. Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck"
        
        elif prediction == 2:
            info = "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. They are firm and often feel like a stone under the skin. "
        
        elif prediction == 3:
            info = "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely inside your body, such as in your nose or throat. The exact cause of all melanomas isn't clear, but exposure to ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases your risk of developing melanoma."
        
        elif prediction == 4:
            info = "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with 'melanocytic nevus', but there are also sources that equate the term mole with any nevus form. A nevus is usually dark and may be raised from the skin."
        
        elif prediction == 5:
            info= "the most common type of benign skin lesion."

        elif prediction == 6:
            info = "A seborrheic keratosis is a common benign skin growth, similar to a mole. Most people will have at least one in their lifetime. They tend to appear in mid-adulthood and their frequency increases with age. They are harmless and don't require treatment, but you can have them removed if they bother you."
            
        elif prediction == 7:
            info = "Squamous cell carcinoma of the skin is caused by the cumulative exposure of the skin to UV light. is the second most common form of skin cancer in the United States"
       
        elif prediction == 8:
            info = "Vascular lesions are abnormal growths or malformations in the blood vessels, which can occur in various parts of the body. They can be congenital or acquired and may result from injury, infection, or other underlying medical conditions. Vascular lesions can range from harmless to potentially life-threatening, depending on their location and severity."

        return render_template('result.html', predicted=predicted_class, image_url=image_url,predicted_class=predicted_class,info=info)

    return render_template('result.html', prediction='Invalid file format', image_url=None)

if __name__ == "__main__":
    app.run()
