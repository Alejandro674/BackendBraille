# Contenido de: tesis_ocr.py (para depuración)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os 

# ---DICCIONARIO--- (Tu código)
braille_dict = {
    'a': [[1, 0], [0, 0], [0, 0]],
    'b': [[1, 0], [1, 0], [0, 0]],
    'c': [[1, 1], [0, 0], [0, 0]],
    'd': [[1, 1], [0, 1], [0, 0]],
    'e': [[1, 0], [0, 1], [0, 0]],
    'f': [[1, 1], [1, 0], [0, 0]],
    'g': [[1, 1], [1, 1], [0, 0]],
    'h': [[1, 0], [1, 1], [0, 0]],
    'i': [[0, 1], [1, 0], [0, 0]],
    'j': [[0, 1], [1, 1], [0, 0]],
    'k': [[1, 0], [0, 0], [1, 0]],
    'l': [[1, 0], [1, 0], [1, 0]],
    'm': [[1, 1], [0, 0], [1, 0]],
    'n': [[1, 1], [0, 1], [1, 0]],
    'o': [[1, 0], [0, 1], [1, 0]],
    'p': [[1, 1], [1, 0], [1, 0]],
    'q': [[1, 1], [1, 1], [1, 0]],
    'r': [[1, 0], [1, 1], [1, 0]],
    's': [[0, 1], [1, 0], [1, 0]],
    't': [[0, 1], [1, 1], [1, 0]],
    'u': [[1, 0], [0, 0], [1, 1]],
    'v': [[1, 0], [1, 0], [1, 1]],
    'w': [[0, 1], [1, 1], [0, 1]],
    'x': [[1, 1], [0, 0], [1, 1]],
    'y': [[1, 1], [0, 1], [1, 1]],
    'z': [[1, 0], [0, 1], [1, 1]],
    'ñ': [[1, 1], [1, 1], [0, 1]],
}

reemplazos = { 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n' }

def reemplazar_caracteres_especiales(texto):
    return ''.join(reemplazos.get(c, c) for c in texto)

# --- (Todo tu código de detPuntos, PVertical, etc. va aquí) ---
# (Omitido por brevedad, pero es el mismo que me diste)

def detPuntos(imagen, diametro_fijo=5, eps=10):
    imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
    bordes = cv2.Canny(imagen_suavizada, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros = []
    for contorno in contornos:
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centros.append([cx, cy])
    if not centros:
        return np.ones_like(imagen) * 255, []
    centros = np.array(centros)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centros)
    etiquetas = clustering.labels_
    nuevos_centros = []
    for etiqueta in np.unique(etiquetas):
        puntos_del_grupo = centros[etiquetas == etiqueta]
        centroide = np.mean(puntos_del_grupo, axis=0).astype(int)
        nuevos_centros.append(centroide)
    nueva_imagen = np.ones_like(imagen) * 255
    radio_fijo = diametro_fijo // 2
    for centroide in nuevos_centros:
        cx, cy = centroide
        cv2.circle(nueva_imagen, (cx, cy), radio_fijo, (0, 0, 0), -1)
    return nueva_imagen, nuevos_centros

def PVertical(celda):
    pixelesNegrosVertical = np.sum(celda == 0, axis=0)
    x = np.arange(pixelesNegrosVertical.shape[0])
    G_vertical = 0.11 * np.max(pixelesNegrosVertical)
    pixelesNegrosVertical[pixelesNegrosVertical <= G_vertical] = 0
    return pixelesNegrosVertical

def convertir_a_matriz_binaria(sub_celda):
    filas, columnas = sub_celda.shape
    if filas == 0 or columnas == 0:
        return np.zeros((3, 2), dtype=int)
    matriz_binaria = np.zeros((3, 2), dtype=int)
    h_fila = filas // 3
    w_columna = columnas // 2
    for i in range(3):
        for j in range(2):
            y1 = i * h_fila
            y2 = (i + 1) * h_fila
            x1 = j * w_columna
            x2 = (j + 1) * w_columna
            cuadrante = sub_celda[y1:y2, x1:x2]
            if np.any(cuadrante == 0):
                matriz_binaria[i, j] = 1
    return matriz_binaria

def identificar_letra(matriz_binaria):
    for letra, matriz in braille_dict.items():
        if np.array_equal(matriz_binaria, np.array(matriz)):
            return letra
    return None

def Braille(rimg, margen=5, umbral_espacio=20, margen_derecho_adicional=20):
    imagen = cv2.imread(rimg, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen desde {rimg}")
        return "Error al cargar imagen", "Error"
    nueva_imagen, puntos_detectados = detPuntos(imagen)
    alto, ancho = nueva_imagen.shape
    imagen_blanca = np.full((alto, ancho), 255, dtype=np.uint8)
    pixelesNegros = np.sum(nueva_imagen == 0, axis=1)
    y = np.arange(pixelesNegros.shape[0])
    G = 0.11 * np.max(pixelesNegros)
    pixelesNegros[pixelesNegros <= G] = 0
    maximos = []
    fila_actual = 0
    for idx, frecuencia in enumerate(pixelesNegros):
        if frecuencia > 0:
            if fila_actual == 0:
                inicio_maximo = idx
            fila_actual += 1
        else:
            if fila_actual > 0:
                maximos.append((inicio_maximo, inicio_maximo + fila_actual - 1))
                fila_actual = 0
    if fila_actual > 0:
        maximos.append((inicio_maximo, inicio_maximo + fila_actual - 1))
    celdas = []
    for i in range(0, len(maximos), 3):
        if i + 2 < len(maximos):
            y1 = max(0, maximos[i][0] - margen)
            y2 = min(nueva_imagen.shape[0], maximos[i + 2][1] + margen)
            celda = nueva_imagen[y1:y2, :]
            celdas.append((celda, y1, y2))
    imagen_final = cv2.cvtColor(nueva_imagen, cv2.COLOR_GRAY2BGR)
    imagen_final_blanca = cv2.cvtColor(imagen_blanca, cv2.COLOR_GRAY2BGR)
    mensaje_por_celdas = []
    mensaje_sin_espacios = []
    for idx, (celda, y1, y2) in enumerate(celdas):
        pixelesNegrosVertical = PVertical(celda)
        maximos_vertical = []
        columna_actual = 0
        for idy, frecuencia in enumerate(pixelesNegrosVertical):
            if frecuencia > 0:
                if columna_actual == 0:
                    inicio_maximo = idy
                columna_actual += 1
            else:
                if columna_actual > 0:
                    maximos_vertical.append((inicio_maximo, inicio_maximo + columna_actual - 1))
                    columna_actual = 0
        if columna_actual > 0:
            maximos_vertical.append((inicio_maximo, inicio_maximo + columna_actual - 1))
        sub_celdas = []
        distancias = []
        j = 0
        umbraladoEspacio = 30
        espacio_entre_columnas = 0
        while j < len(maximos_vertical):
            if j + 1 < len(maximos_vertical):
                espacio_entre_columnas = maximos_vertical[j + 1][0] - maximos_vertical[j][1]
                if espacio_entre_columnas > umbral_espacio:
                    x1 = max(0, maximos_vertical[j][0] - margen)
                    x2 = min(celda.shape[1], maximos_vertical[j][1] + margen + margen_derecho_adicional)
                    sub_celda = celda[:, x1:x2]
                    sub_celdas.append((sub_celda, x1, x2))
                    distancias.append(x2)
                    j += 1
                else:
                    x1 = max(0, maximos_vertical[j][0] - margen)  
                    x2 = min(celda.shape[1], maximos_vertical[j + 1][1] + margen) 
                    sub_celda = celda[:, x1:x2]
                    sub_celdas.append((sub_celda, x1, x2))
                    distancias.append(x2)
                    j += 2
            else:
                x1 = max(0, maximos_vertical[j][0] - margen)
                x2 = min(celda.shape[1], maximos_vertical[j][1] + margen + margen_derecho_adicional)
                sub_celda = celda[:, x1:x2]
                sub_celdas.append((sub_celda, x1, x2))
                distancias.append(x2)
                j += 1
        distancias_entre_subceldas = np.zeros((len(sub_celdas) - 1, 1), dtype=int)
        for i in range(len(sub_celdas) - 1):
            if sub_celdas[i][2] < sub_celdas[i+1][1]:
                distancia_entre_subceldas = sub_celdas[i + 1][1] - sub_celdas[i][2]
                distancias_entre_subceldas[i] = distancia_entre_subceldas
        letras_celda = []
        for i, (sub_celda, x1, x2) in enumerate(sub_celdas):
            matriz_binaria = convertir_a_matriz_binaria(sub_celda)
            letra = identificar_letra(matriz_binaria)
            if letra:
                letras_celda.append((letra, x1, y1))
                if i < len(distancias_entre_subceldas) and distancias_entre_subceldas[i] >= umbraladoEspacio:
                    letras_celda.append((' ', x2, y1))
        mensaje_por_celdas.append(" ".join([letra for letra, x1, y1 in letras_celda]))
        mensaje_sin_espacios.append("".join([letra for letra, x1, y1 in letras_celda]))
    mensaje = "\n".join(mensaje_por_celdas)
    mensajeSE = "\n".join(mensaje_sin_espacios)
    print("Mensaje en Braille:", mensaje)
    cv2.imwrite("Traduccion.png", imagen_final_blanca)
    return mensaje, mensajeSE

# --- NUESTRA NUEVA FUNCIÓN "ENVOLTORIO" (WRAPPER) ---
def procesar_imagen_tesis(imagen_cv):
    print("[MODULO TESIS] Imagen recibida en memoria.")
    ruta_temporal = "temp_image_from_api.jpg"
    
    try:
        # 1. Guarda la imagen que nos dio la API en un archivo
        cv2.imwrite(ruta_temporal, imagen_cv)
        print(f"[MODULO TESIS] Imagen guardada en {ruta_temporal}")

        # 2. ¡Llama a tu código de tesis con esa ruta!
        traduccion, mensaje_hablado = Braille(ruta_temporal)
        
        # --- ¡CAMBIO! NO BORRAMOS LA IMAGEN ---
        # if os.path.exists(ruta_temporal):
        #     os.remove(ruta_temporal)
        # --------------------------------------
            
        print(f"[MODULO TESIS] Traducción exitosa: {traduccion}")
        return traduccion, mensaje_hablado

    except Exception as e:
        print(f"[MODULO TESIS] ¡Error al procesar! {e}")
        # --- ¡CAMBIO! NO BORRAMOS LA IMAGEN ---
        # if os.path.exists(ruta_temporal):
        #     os.remove(ruta_temporal)
        # --------------------------------------
        return f"Error en el módulo de IA: {e}", "Error"
