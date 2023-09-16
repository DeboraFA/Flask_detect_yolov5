from flask import Flask, render_template, request, jsonify, session, redirect, make_response, send_file
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
import base64
import pdfkit
import numpy as np
import onnxruntime as ort
import cv2
import PIL
import pickle
from datetime import datetime
from yolo_predictions import YOLO_Pred
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from glaucoma_segmentation import test_segmentation_disc, test_segmentation_cup, CDR, BVR, NRR

import os 
from fpdf import FPDF

app = Flask(__name__)

# Chave secreta para usar a sessão
app.secret_key = 'chave_secreta'

# Carrega o modelo YOLO
onnx_model = 'model/yolov5.onnx'
data_yaml = 'model/data.yaml'
yolo_pred = YOLO_Pred(onnx_model, data_yaml)

seg_encoder = 'mobilenet_v2' # https://github.com/qubvel/segmentation_models.pytorch
seg_model = 'Unetplusplus' #  Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3,DeepLabV3Plus




import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image



def classification(img):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    for layer in vgg.layers:
        layer.trainable = False
    output = vgg.layers[-1].output
    output = Flatten()(output)
    vgg_model = Model(vgg.input, output)
    # vgg_model.summary()
    # img = Image.open(img)  # Use OpenCV to read the image
    x = image.img_to_array(img)
    x = cv2.resize(x, (224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = vgg_model.predict(x)
    flat = feature.flatten()
    pickled_model = pickle.load(open('model/classificacao/vgg16_SVM_original.sav', 'rb'))
    y_pred = pickled_model.predict(flat.reshape(1, -1))
    if int(y_pred)==0:
        pred = 'Normal'
    elif int(y_pred)==1:
        pred = 'Glaucoma'

    return pred

# Variável de controle para o envio de imagens
image_processing_active = True
counter = 1


@app.route('/formulario', methods=['GET', 'POST'])
def formulario():
    enviado = False
    if request.method == 'POST':
        nome = request.form['nome']
        cpf = request.form['cpf']
        pasta = criar_pasta_e_pdf(nome, cpf)  # Obter o caminho completo da pasta
        enviado = True

        # Armazena o nome da pasta na sessão
        session['nome_pasta'] = pasta
        global image_processing_active
        image_processing_active = True  # Reset the flag here

    return render_template('formulario.html', enviado=enviado)

def criar_pasta_e_pdf(nome, cpf):
    base_pasta = f'static/formularios/{nome}_{cpf}'
    
    # Verifica se a pasta já existe
    if os.path.exists(base_pasta):
        # Obter a data e hora atual
        now = datetime.now()
        data_hora_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Criar uma nova pasta com nome base_pasta + data_hora_str
        pasta = f'{base_pasta}/{nome}_{cpf}_{data_hora_str}'
        os.makedirs(pasta)
    else:
        # Se a pasta não existe, crie normalmente
        pasta = base_pasta
        os.makedirs(pasta)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, f'Nome: {nome}')
    pdf.cell(40, 10, f'CPF: {cpf}')
    pdf.output(f'{pasta}/dados.pdf', 'F')
    return pasta 


@app.route('/')
def inicio():
    session.pop('nome_pasta', None)
    return render_template('inicio.html')

@app.route('/avaliar', methods=['GET'])
def avaliar():
    if 'nome_pasta' not in session:
        global image_processing_active
        image_processing_active = True  # Set the flag to True if session data is not available
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global image_processing_active, counter
    # Verifica se o processamento de imagem está ativo
    if not image_processing_active:
        print('parooooooooooou')
        return jsonify({'message': 'Image processing stopped'})

    # Obtém o nome da pasta armazenado na sessão
    pasta = session.get('nome_pasta')
    if pasta is None:
        return jsonify({'message': 'Session data not available'})
    
    # Extrai a imagem enviada pelo cliente
    # image_data = request.json['image']
    image_data = request.json.get('image', '')

    # Verifica se o valor de image_data é uma string em base64 válida
    if not base64.b64decode(image_data, validate=True):
        return jsonify({'message': 'Invalid base64 string'})

    try:
        # Converte a imagem base64 para array numpy
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (640, 640))
        image2 = image.copy()

        
        # Faz a predição da imagem
        bb_conf, boxes, index0 = yolo_pred.predictions(image)
        print(f'aquiiiiiiiii________{bb_conf}')
        if bb_conf >= 70 and counter <=3:
                boxes_np = np.array(boxes).tolist()
                print(f'------------nova imagem_{counter}------------')
                print(len(boxes_np))
                print(np.shape(boxes_np))
                print('-----------------------------------')
                # Desenha retângulos nos objetos detectados na imagem
                for ind in index0:
                    # print(counter)
                    x, y, w, h = boxes_np[ind]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Desenha o retângulo na imagem usando OpenCV

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if len(boxes_np)<50:
                    # Salva a imagem com os objetos detectados
                    cv2.imwrite(f'{pasta}/imagem_corte_{counter}.png', image2[y-15:y+h+15,x-15:x+w+15])
                    # cv2.imwrite(f'{pasta}/imagem_{counter}.png', image)
                    

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
    
@app.route('/reiniciar', methods=['GET'])
def reiniciar():
    session.pop('nome_pasta', None)
    global image_processing_active, counter
    image_processing_active = False
    counter = 1
    return render_template('inicio.html')

    
@app.route('/visualizar_resultados', methods=['GET'])
def visualizar_resultados():
    pasta = session.get('nome_pasta')
    if pasta is None:
        return jsonify({'message': 'Session data not available'})

    imagens = []
    for i in range(1, counter):
        imagem_path = f'{pasta}/imagem_corte_{i}.png'
        if os.path.exists(imagem_path):
            imagens.append(imagem_path)

    # Verifique se existem imagens na pasta e defina uma variável de controle
    botao_visivel = len(imagens) > 0

    return render_template('visualizar.html', imagens=imagens, botao_visivel=botao_visivel)


image_files = []
info_lines = []
segmented_image_paths = []
cdrs = []
bvrs = []
nrrs = []
preds = []

@app.route('/resultados')
def imagens_exemplo():
    global image_files, info_lines, segmented_image_paths, cdrs, bvrs, nrrs, preds

    # image_folder = 'static/formularios/Debora_1234'
    # Obtém o nome da pasta armazenado na sessão
    image_folder = session.get('nome_pasta')
    if image_folder is None:
        return jsonify({'message': 'Session data not available'})
    # image_files = ['imagem_corte_1.png', 'imagem_corte_2.png', 'imagem_corte_3.png']  # Add more images as needed
    image_files = [file for file in os.listdir(image_folder) if file.endswith('.png')]



    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image_test = Image.open(image_path)  # Use OpenCV to read the image

        

        image = np.array(image_test)

        contours_cup = test_segmentation_cup(image_test, seg_encoder, seg_model)
        contours_disc = test_segmentation_disc(image_test, seg_encoder, seg_model)


        print('abriiiiiiiiiiiiiiiuuuuuuuu')

        result = image.copy()
        result = cv2.resize(result, (224,224))
        

        for cnt_disc in contours_disc:
            img_contours1 = cv2.drawContours(result, [cnt_disc], -1, (0, 255, 0), 2)

        for cnt_cup in contours_cup:
            img_contours = cv2.drawContours(result, [cnt_cup], -1, (255, 0, 0), 2)

            cdr = CDR(cnt_disc, cnt_cup)
            bvr = BVR(cnt_disc, image)
            nrr = NRR(cnt_disc, cnt_cup, image)
            predito = classification(image_test)

            pil_result = Image.fromarray(img_contours)
            pil_result_resized = pil_result.resize((224, 224))  # Resize using PIL

            seg_folder = image_folder + '/' + 'result_seg'
            if not os.path.exists(seg_folder):
                os.makedirs(seg_folder)


            segmented_image_path = f"{seg_folder}/segmented_image_{len(segmented_image_paths)}.png"
            pil_result_resized.save(segmented_image_path, format="PNG")

            segmented_image_paths.append(segmented_image_path)
            cdrs.append(cdr)
            bvrs.append(bvr)
            nrrs.append(nrr)
            preds.append(predito)
        
            


    info_lines = [
        "Informação 1 da Imagem",
        "Informação 2 da Imagem",
        "Informação 3 da Imagem",
        "Informação 4 da Imagem"
    ]


    rendered_html = render_template(
        'resultados.html',
        image_folder=image_folder,
        image_files=image_files,
        info_lines=info_lines,
        segmented_image_paths=segmented_image_paths,
        cdrs=cdrs,
        bvrs=bvrs,
        nrrs=nrrs,
        preds=preds
    )

    segmented_image_paths = []
    return rendered_html


@app.route('/download-pdf', methods=['POST'])
def download_pdf():

    global info_lines, segmented_image_paths, cdrs, bvrs, nrrs, preds
    
    image_folder = session.get('nome_pasta')
    if image_folder is None:
        return jsonify({'message': 'Session data not available'})

    # O código para obter os resultados em HTML permanece o mesmo
    rendered_html = render_template(
        'resultados.html',
        image_folder=image_folder,
        image_files=image_files,
        info_lines=info_lines,
        segmented_image_paths=segmented_image_paths,
        cdrs=cdrs,
        bvrs=bvrs,
        nrrs=nrrs,
        preds=preds
    )

    pdfkit_config = pdfkit.configuration(wkhtmltopdf='C:/Users/debora.assis/Anaconda3/Lib/site-packages/wkhtmltopdf')
    pdfkit.from_file('resultados.html', 'output.pdf', configuration=pdfkit_config)

    # Crie um arquivo PDF a partir do HTML renderizado
    pdf_file_path = 'resultados.pdf'  # Substitua pelo caminho desejado
    pdfkit.from_string(rendered_html, pdf_file_path)

    # Retorne o PDF como uma resposta para download
    response = make_response(send_file(pdf_file_path, as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment; filename=resultados.pdf'

    return response

@app.route('/reiniciar_exame', methods=['GET'])
def reiniciar_exame():
    # Obtém o nome da pasta armazenado na sessão
    pasta = session.get('nome_pasta')
    if pasta is None:
        return jsonify({'message': 'Session data not available'})
    
    global image_processing_active, counter
    image_processing_active = True  # Defina como True para permitir um novo exame
    counter = 1  # Reinicie a contagem para a nova série de imagens

    # Deleta as imagens antigas para sobrepor com as novas
    for i in range(1, 4):  # Substitua 4 pelo número máximo de imagens que você deseja capturar
        imagem_path = f'{pasta}/imagem_corte_{i}.png'
        if os.path.exists(imagem_path):
            os.remove(imagem_path)

    return redirect('/avaliar')



@app.route('/stop', methods=['POST'])
def stop():
    global image_processing_active
    image_processing_active = False
    return jsonify({'message': 'Image processing stopped'})



if __name__ == '__main__':
    app.run(debug=True)
