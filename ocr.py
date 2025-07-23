import easyocr
import pytesseract
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

easy_ocr = easyocr.Reader(['en'])
@app.route('/easy-ocr', methods=['POST'])
def easy_ocr_method():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    results = easy_ocr.readtext(img)
    return jsonify([result[1] for result in results])


paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
@app.route('/paddle-ocr', methods=['POST'])
def paddle_ocr_method():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    results = paddle_ocr.predict(img)
    text_results = [line[1][0] for line in results[0]]
    return jsonify(text_results)


@app.route('/tesseract-ocr', methods=['POST'])
def tesseract_ocr_method():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb)
    return jsonify({'text': text})


trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
@app.route('/transformer-ocr', methods=['POST'])
def transformer_ocr_method():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    pixel_values = trocr_processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({"text": text})


if __name__ == '__main__':
    app.run(port=5001)
