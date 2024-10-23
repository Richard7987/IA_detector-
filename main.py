import cv2
import cvzone
from collections import deque

thres = 0.55  # Umbral de confianza
nmsThres = 0.2  # Umbral de NMS (Non-Maxima Suppression)

cap = cv2.VideoCapture('v1.mp4')
cap.set(3, 640)
cap.set(4, 480)

# Carga los nombres de las clases
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().strip().split('\n')

# Cargar la configuración y los pesos de la red
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Inicializa el contador de personas y una lista de seguimiento
person_count = 0
trackers = []  # Lista de trackers activos
max_distance = 50  # Distancia máxima entre dos detecciones para considerar que es la misma persona
detections = deque(maxlen=20)  # Cola para almacenar las últimas detecciones de personas

while True:
    success, img = cap.read()
    if not success:
        break  # Sale del bucle si no se pueden leer más frames

    # Detección de objetos
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    # Verifica si la detección no está vacía
    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 < classId <= len(classNames):
                if classNames[classId - 1].lower() == "person":
                    # Verifica si la persona ya fue contada en los últimos frames
                    x, y, w, h = box
                    detected = False

                    for prev_box in detections:
                        prev_x, prev_y, prev_w, prev_h = prev_box
                        distance = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                        if distance < max_distance:
                            detected = True
                            break
                    
                    if not detected:
                        person_count += 1  # Incrementa el contador solo si no ha sido detectada recientemente
                        detections.append(box)  # Agrega la nueva detección a la cola

                    # Dibuja el cuadro alrededor de la persona
                    cvzone.cornerRect(img, box)
                    cv2.putText(img, f'PERSON {round(conf * 100, 2)}%',
                                (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Al finalizar el video, muestra el conteo de personas
print(f'Cantidad total de personas detectadas: {person_count}')
