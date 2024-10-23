# 1- LES IMPORTS
# Import d'OpenCV
import time
import cv2
import numpy as np

# 2- YOLO

# Chargement de YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Chargement des noms des classes depuis le fichier coco
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# On récupère la couche de sortie
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# On affecte une couleur aléatoire
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Index de la classe "chien" dans le fichier coco.names (généralement 12 pour les chiens)
dog_class_id = classes.index("dog")

# Affichage

cap = cv2.VideoCapture(0)  # mettre 0 pour caméra par défaut ou flux vidéo quelconque

while True:
    # On prend la photo
    ret, frame = cap.read()
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Détection d'objets
    blob = cv2.dnn.blobFromImage(img, 0.00392, (640, 480), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Propagation avant
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.01 and class_id == dog_class_id:  # Condition pour détecter uniquement les chiens
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Coordonnées du rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Dessin des rectangles autour des chiens détectés
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("TITRE DE MA FENETRE", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
