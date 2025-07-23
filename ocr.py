import easyocr
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
reader = easyocr.Reader(['en'])

@app.route('/ocr', methods=['POST'])
def ocr():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    results = reader.readtext(img)
    return jsonify([result[1] for result in results])

if __name__ == '__main__':
    app.run(port=5001)
