import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from flask import Flask, request, jsonify, render_template
import json

import io
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
import torch
import onnxruntime as ort

app = Flask(__name__)

# Carrega o modelo ONNX
session = ort.InferenceSession('model/yolov5.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
num_classes = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Extrai a imagem enviada pelo cliente
    image_data = request.json['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocessa a imagem
    image = image.resize((640, 640))
    img2 = image.copy()
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    image_tensor = torch.from_numpy(image_np)

    # Faz a inferência na imagem
    output = session.run([output_name], {input_name: image_np})    
    output = np.squeeze(output)

    # Filtra detecções com confiança menor que um limiar
    threshold = 0.7
    class_probs = torch.sigmoid(torch.from_numpy(output[:, 4:5]))
    filter_mask = class_probs > threshold
    class_probs = class_probs[filter_mask].numpy()
    boxes = output[filter_mask.numpy().squeeze(), :4]
    print(boxes.size)
    if boxes.size > 0:
        for box in boxes:
            
            x, y, w, h = box
            # print(f'x: {x}')
            # Converte as coordenadas normalizadas para coordenadas de pixel
            image_width, image_height = image.size
            x1 = int(x * image_width)
            y1 = int(y * image_height)
            x2 = int((x + w) * image_width)
            y2 = int((y + h) * image_height)

            # Desenha retângulos nos objetos detectados
            draw = ImageDraw.Draw(img2)
            # Desenha o retângulo na imagem
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
        # Salva a imagem com as detecções
        img2.save("exemplo.png") 
    
    # Codifica a imagem de volta para base64
    buffered = io.BytesIO()
    img2.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Retorna a imagem com os objetos detectados em formato JSON
    response_data = {'image': img_str}
    return json.dumps(response_data)

if __name__ == '__main__':
    app.run(debug=True)
