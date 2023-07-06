from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import onnxruntime as ort
import torch
import re
import cv2
import PIL
from yolo_predictions import YOLO_Pred

app = Flask(__name__)

# Carrega o modelo YOLO
onnx_model = 'model/yolov5.onnx'
data_yaml = 'model/data.yaml'
yolo_pred = YOLO_Pred(onnx_model, data_yaml)

# Variável de controle para o envio de imagens
image_processing_active = True
counter = 1
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global image_processing_active, counter
    # Verifica se o processamento de imagem está ativo
    if not image_processing_active:
        print('parooooooooooou')
        return jsonify({'message': 'Image processing stopped'})
    
    # Extrai a imagem enviada pelo cliente
    image_data = request.json['image']
    # print('aaaaaaaaaaaaaaaaaaa')
    # Verifica se o valor de image_data é uma string em base64 válida
    if not base64.b64decode(image_data, validate=True):
        return jsonify({'message': 'Invalid base64 string'})

    try:
        # Converte a imagem base64 para array numpy
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (640, 640))

        
        # Faz a predição da imagem
        bb_conf, boxes, index = yolo_pred.predictions(image)
        # print(f'aquiiiiiiiii________{bb_conf}')
        if bb_conf >= 60:
            boxes_np = np.array(boxes).tolist()
            print(f'------------nova imagem_{counter}------------')
            print(len(boxes_np))
            print(np.shape(boxes_np))
            print('-----------------------------------')
            # Desenha retângulos nos objetos detectados na imagem
            for ind in index:
                # print(counter)
                x, y, w, h = boxes_np[ind]
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Desenha o retângulo na imagem usando OpenCV
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if len(boxes_np)<50:
                # Salva a imagem com os objetos detectados
                cv2.imwrite(f'exemplo_{counter}.png', image)
                # Incrementa o contador
                counter += 1

        # Codifica a imagem de volta para base64
        _, encoded_image = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(encoded_image).decode('utf-8')

        # Retorna a imagem com os objetos detectados em formato JSON
        response_data = {'image': img_str}
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop():
    global image_processing_active
    image_processing_active = False
    return jsonify({'message': 'Image processing stopped'})

if __name__ == '__main__':
    app.run(debug=True)
