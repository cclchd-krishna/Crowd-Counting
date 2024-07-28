from flask import Flask, request, render_template, send_file, jsonify, url_for
import cv2
import numpy as np
import os
from pathlib import Path
import requests
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# File paths
yolov4_cfg_path = "yolov4.cfg"
yolov4_weights_path = "yolov4.weights"
coco_names_path = "coco.names"


# Load YOLO
net = cv2.dnn.readNet(yolov4_weights_path, yolov4_cfg_path)
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file", 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            height, width, channels = image.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            person_count = 0
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if label == "person":
                        person_count += 1
                    color = (0, 255, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y + 30), font, 1, color, 2)

            processed_filename = 'processed_' + filename
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, image)

            image_url = url_for('static', filename='uploads/' + processed_filename)
            
            return render_template('index.html', image_url=image_url, person_count=person_count)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
