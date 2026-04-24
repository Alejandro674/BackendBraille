from flask import Flask, request, jsonify, send_file
import base64
import cv2
import numpy as np
import os
import docx
import PyPDF2
import io
import re 

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import tesis_ocr
from symspellpy import SymSpell, Verbosity

app = Flask(__name__)

sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = "es_50k.txt"
try:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    print("Diccionario de IA en español (es_50k.txt) cargado exitosamente.")
except Exception as e:
    print(f"¡ERROR! No se pudo cargar el diccionario de IA: {e}")
    print("Asegúrate de haber descargado 'es_50k.txt' y puesto en la carpeta BrailleAPI.")

char_to_pattern = {
    'A': '100000', 'B': '110000', 'C': '100100', 'D': '100110', 'E': '100010',
    'F': '110100', 'G': '110110', 'H': '110010', 'I': '010100', 'J': '010110',
    'K': '101000', 'L': '111000', 'M': '101100', 'N': '101110', 'O': '101010',
    'P': '111100', 'Q': '111110', 'R': '111010', 'S': '011100', 'T': '011110',
    'U': '101001', 'V': '111001', 'W': '010111', 'X': '101101', 'Y': '101111',
    'Z': '101011',
    ' ': '000000',
    
}

pattern_to_unicode = {
    '100000': '⠁', '110000': '⠃', '100100': '⠉', '100110': '⠙', '100010': '⠑',
    '110100': '⠋', '110110': '⠛', '110010': '⠓', '010100': '⠊', '010110': '⠚',
    '101000': '⠅', '111000': '⠇', '101100': '⠍', '101110': '⠝', '101010': '⠕',
    '111100': '⠏', '111110': '⠟', '111010': '⠗', '011100': '⠎', '011110': '⠞',
    '101001': '⠥', '111001': '⠧', '010111': '⠺', '101101': '⠭', '101111': '⠽',
    '101011': '⠵',
    '000000': ' ', # Espacio
    '111011': '⠷', # á
    '011101': '⠮', # é
    '001100': '⠌', # í
    '001101': '⠬', # ó
    '011111': '⠾', # ú
    '010000': '₉', # Coma
    '010010': '⠒', # Punto / Dos Puntos
    '010001': '⠢', # Interrogación
    '011010': '⠖', # Exclamación
}

@app.route("/api/detect", methods=['POST'])
def detect_braille():
    print("\n[Ruta /detect] Petición de Cámara recibida...")
    try:
        data = request.get_json()
        image_b64 = data['image']
        crop_params = data['crop_params']
        
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv_original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
        original_image_width = crop_params['original_image_width']
        original_image_height = crop_params['original_image_height']
        screen_width = crop_params['screen_width']
        screen_height = crop_params['screen_height']
        scale_x = original_image_width / screen_width
        scale_y = original_image_height / screen_height
        
        x_start = int(crop_params['x'] * scale_x)
        y_start = int(crop_params['y'] * scale_y)
        width = int(crop_params['width'] * scale_x)
        height = int(crop_params['height'] * scale_y)
        
        img_cv = img_cv_original[y_start:y_start+height, x_start:x_start+width]

        if img_cv.shape[0] == 0 or img_cv.shape[1] == 0:
            raise ValueError("La imagen recortada está vacía.")
        
        print("Imagen recortada, usando lógica de tesis para 1 letra...")
        
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        nueva_imagen, puntos_detectados = tesis_ocr.detPuntos(img_gray)
        matriz_binaria = tesis_ocr.convertir_a_matriz_binaria(nueva_imagen)
        letra_detectada = tesis_ocr.identificar_letra(matriz_binaria)
        
        if letra_detectada is None:
            letra_detectada = "?"

        print(f"[Ruta /detect] Letra final detectada: {letra_detectada}")
        
    except Exception as e:
        print(f"Error en /detect: {e}")
        return jsonify({"success": False, "error": str(e), "letra": "?"})
    
    return jsonify({"success": True, "letra": letra_detectada})


@app.route("/api/extract_text_from_doc", methods=['POST'])
def extract_text_from_doc():
    print("\n[Ruta /extract_text_from_doc] Petición recibida...")
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No se envió ningún archivo."})

        file = request.files['file']
        text = "" 
        
        if file.mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = docx.Document(file)
            fullText = [para.text for para in document.paragraphs]
            text = '\n'.join(fullText)
        elif file.mimetype == 'application/pdf':
            pdfReader = PyPDF2.PdfReader(file)
            fullText = [page.extract_text() for page in pdfReader.pages]
            text = '\n'.join(fullText)
        else:
            return jsonify({"success": False, "error": f"Tipo de archivo no soportado: {file.mimetype}"})
        
        print(f"Texto extraído: {text[:50]}...")
        if not text.strip():
             return jsonify({"success": False, "error": "El documento está vacío o no se pudo leer texto."})

        braille_string = ""
        for char in text:
            pattern = char_to_pattern.get(char.upper(), '000000') 
            braille_char = pattern_to_unicode.get(pattern, ' ')
            braille_string += braille_char

        print(f"Texto Braille generado: {braille_string[:50]}...")
        return jsonify({"success": True, "braille_text": braille_string})

    except Exception as e:
        print(f"Error en /extract_text_from_doc: {e}")
        return jsonify({"success": False, "error": str(e)})


def draw_braille_pdf_cell(c, pattern, x, y, dot_radius=2.5, dot_gap=8):
    dots = list(pattern)
    p1 = (x + dot_gap, y + (2 * dot_gap) + (2 * dot_radius))
    p2 = (x + dot_gap, y + dot_gap + dot_radius)
    p3 = (x + dot_gap, y)
    p4 = (x + (2 * dot_gap) + dot_radius, y + (2 * dot_gap) + (2 * dot_radius))
    p5 = (x + (2 * dot_gap) + dot_radius, y + dot_gap + dot_radius)
    p6 = (x + (2 * dot_gap) + dot_radius, y)
    dot_positions = [p1, p2, p3, p4, p5, p6]
    c.setFillColorRGB(0, 0, 0)
    for i, (px, py) in enumerate(dot_positions):
        if dots[i] == '1':
            c.circle(px, py, dot_radius, fill=1, stroke=0)
        else:
            c.circle(px, py, dot_radius / 2, fill=0, stroke=1)
    return (3 * dot_gap) + (2 * dot_radius)

@app.route("/api/generate_braille_pdf", methods=['POST'])
def generate_braille_pdf():
    print("\n[Ruta /generate_braille_pdf] Petición recibida...")
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"success": False, "error": "No se recibió texto."})
        
        text = data['text']
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        margin = 1 * inch
        x = margin
        y = height - margin
        
        c.setFont("Helvetica", 14) 
        text_object = c.beginText(x, y)
        text_object.setTextOrigin(x, y)
        
        lines = text.split('\n')
        for line in lines:
            text_object.textLine(line)
            
        c.drawText(text_object)
        c.save()
        
        buffer.seek(0)
        print("PDF (simple) generado, enviando archivo...")
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="traduccion_braille.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        print(f"Error en /generate_braille_pdf: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/detect_braille_image", methods=['POST'])
def detect_braille_image():
    print("\n[Ruta /detect_braille_image] Petición recibida...")
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"success": False, "error": "No se envió imagen en base64."})
        
        image_b64 = data['image']
        
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        print("Imagen decodificada, llamando al módulo de tesis...")

        traduccion, texto_para_hablar = tesis_ocr.procesar_imagen_tesis(img_cv)
        
        print(f"Texto de tesis (junto): {texto_para_hablar[:100]}...")
        
        texto_junto_limpio = texto_para_hablar.replace('\n', ' ').replace('\r', ' ')
        sugerencias = sym_spell.word_segmentation(texto_junto_limpio, max_edit_distance=0)
        texto_final_corregido = sugerencias.corrected_string
        
        print(f"Texto corregido por IA (SymSpell): {texto_final_corregido[:100]}...")

        return jsonify({
            "success": True, 
            "text": texto_final_corregido,
            "speech_text": texto_final_corregido
        })

    except Exception as e:
        print(f"Error en /detect_braille_image: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("Iniciando servidor Flask con IA de segmentación en ESPAÑOL...")
    app.run(debug=True, host='0.0.0.0', port=5000)
