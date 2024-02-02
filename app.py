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
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os.path



app = FastAPI()
functions = []

# ÐšÐ°Ð¼ÐµÑ€Ð° 1
# rtsp://tapocam:neural@213.137.237.165:554/stream2

# ÐšÐ°Ð¼ÐµÑ€Ð° 2
# rtsp://tapocam2:neural@213.137.237.165:555/stream2

# ÐšÐ°Ð¼ÐµÑ€Ð° 3
# rtsp://tapocam3:neural@213.137.237.165:556/stream222
# camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("rtsp://tapocam:neural@213.137.237.165:554/stream1")
templates = Jinja2Templates(directory="templates")

def gen_frames(camera, model):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            for face_encoding in face_encodings:
                closest_distances = model.kneighbors([face_encoding], n_neighbors=1)
                if len(closest_distances[1]) > 0:
                    name = model.classes_[closest_distances[1][0][0]] 
                    print(name)
                else:
                    print("Unknown")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def trainer(ip=''):
    path = './data'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_encodings_list = []
    names_list = []
    for imagePath in imagePaths:
        name = os.path.splitext(os.path.basename(imagePath))[0]
        image = face_recognition.load_image_file(imagePath)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if face_encodings:
            face_encodings_list.append(face_encodings[0])
            names_list.append(name)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(face_encodings_list, names_list)
    with open(f"./trained_face_recognition_model_{ip}.pkl", "wb") as f:
        pickle.dump(model, f)
    return True

if __name__ == '__main__':
    uvicorn.run(app, host='188.120.249.61', port=8000, debug=True)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    camera = cv2.VideoCapture("rtsp://tapocam:neural@213.137.237.165:554/stream1")
    with open("./trained_face_recognition_model.pkl", "rb") as f:
        model = pickle.load(f)
    return StreamingResponse(gen_frames(camera, model), media_type='multipart/x-mixed-replace; boundary=frame')

def create_dynamic_function(ip, add):
    function_body = f"""
@app.get('/video_feed_{ip}')
def video_feed_{ip}():
    if os.path.exists(f"./trained_face_recognition_model_{ip}.pkl"):
        tr = True
    else:
        tr = trainer({ip})
    if tr:
        camera = cv2.VideoCapture("{str(add)}")
        with open("./trained_face_recognition_model_{ip}.pkl", "rb") as f:
            model = pickle.load(f)
        return StreamingResponse(gen_frames(camera, model), media_type='multipart/x-mixed-replace; boundary=frame')
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
    global functions
    print(functions, "____________________")
    if ip in functions:
        pass
    else:
        create_dynamic_function(ip, add)
    return Response(content="IP address received successfully!", status_code=201)
