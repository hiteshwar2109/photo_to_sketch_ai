import numpy as np
import cv2
from flask import Flask, render_template, request
from io import BytesIO
import base64

app = Flask(__name__)

def convert_to_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred_img = 255 - blurred_img
    pencil_sketch = cv2.divide(gray_image, inverted_blurred_img, scale=256.0)
    return pencil_sketch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    if img is None:
        return render_template('index.html', error='Error reading the image')

    sketch = convert_to_sketch(img)

    # Encode the original and sketch images to base64
    _, original_img_encoded = cv2.imencode('.jpg', img)
    _, sketch_img_encoded = cv2.imencode('.jpg', sketch)

    # Convert encoded images to base64 strings
    original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
    sketch_img_base64 = base64.b64encode(sketch_img_encoded).decode('utf-8')

    original_data_url = f"data:image/jpeg;base64,{original_img_base64}"
    sketch_data_url = f"data:image/jpeg;base64,{sketch_img_base64}"

    return render_template('index.html', output=original_data_url, sketch=sketch_data_url)

if __name__ == '__main__':
    app.run(debug=True)
