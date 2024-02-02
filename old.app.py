from fastapi import Request
import cv2
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import os
import json
import ast
import threading


app = FastAPI()

# Камера 1
# rtsp://tapocam:neural@213.137.237.165:554/stream2

# Камера 2
# rtsp://tapocam2:neural@213.137.237.165:555/stream2

# Камера 3
# rtsp://tapocam3:neural@213.137.237.165:556/stream222
# camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("rtsp://tapocam:neural@213.137.237.165:554/stream1")
templates = Jinja2Templates(directory="templates")

def recognizer(gray, x, y, w, h, frame, json_data):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/faces.xml")
    user_ID, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    print(confidence, "==================>", user_ID)
    if confidence < 75:
        filtered_data = [item for item in json_data if item["id"] == user_ID]
        cv2.putText(frame, f"{filtered_data[0]['name']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    elif confidence > 100:
        cv2.putText(frame, "Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    
def gen_frames(camera, face_cascade):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            with open('data.json', "r") as file:
                json_data = json.load(file)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 4)
                recognizer(gray, x, y, w, h, frame, json_data)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    ids = []
    data = []
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        name = (imagePath.split("/")[1]).split('.')[0] 
        id = np.random.randint(1, 100000) 
        data_item = {"id": id, "name": name}
        data.append(data_item)
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    with open("data.json", "w") as file:
        json.dump(data, file)
    return faceSamples,ids

def trainer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces,ids = getImagesAndLabels('data')
    recognizer.train(faces, np.array(ids))
    recognizer.save(f'trainer/faces.xml')
    return True

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/train')
def training(request: Request):
    tr = trainer()
    if tr:
        return Response(content='Successfully Trained', media_type="application/json")
    else:
        return Response(content='Unuccessfully Trained', media_type="application/json")

@app.get('/video_feed')
def video_feed():
    camera = cv2.VideoCapture("rtsp://tapocam:neural@213.137.237.165:554/stream1")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return StreamingResponse(gen_frames(camera, face_cascade), media_type='multipart/x-mixed-replace; boundary=frame')

def create_dynamic_function(ip, add):
    function_body = f"""
@app.get('/video_feed_{ip}')
def video_feed_{ip}():
        camera = cv2.VideoCapture("{str(add)}")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        return StreamingResponse(gen_frames(camera, face_cascade), media_type='multipart/x-mixed-replace; boundary=frame')
    """
    globals().update({f"video_feed_{ip}": None})
    ast_tree = ast.parse(function_body)
    code_object = compile(ast_tree, filename="<ast>", mode="exec")
    exec(code_object, globals())

@app.post('/camera-ip')
async def cameraIp(request: Request):
    add_data = await request.body()
    add = json.loads(add_data.decode("utf-8"))
    add = str(add['ip'])
    ip = (((add.split("@")[1]).split('/')[0]).replace(':','')).replace('.','')
    print(ip)
    print(add)
    create_dynamic_function(ip, add)
    return Response(content="IP address received successfully!", status_code=201)