import cv2, os
from matplotlib import image
import numpy as np
from PIL import Image
from threading import Thread
from datetime import datetime as dt
from multiprocessing import Process
import time
class unknownUsers:
    def __init__(self, img, date_time):
        self.img = img
        self.date_time = date_time

def unknown_user_rec():
    global camera
    global camRec
    _, frame = camera.read()
    time_date = dt.now()
    cv2.imwrite(f"./static/unknown/" + f"{time_date.year}.{time_date.month}.{time_date.day}_{time_date.hour}.{time_date.minute}.{time_date.second}" + ".jpg", frame)
    time.sleep(5)
    camRec = False

def recognizer(gray, x, y, w, h, frame):
    pass
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read("./trainer/faces.xml")
    # user_ID, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    # print(confidence, "==================>")
    # global which_camera
    # if confidence < 75:
    #     cv2.putText(frame, "Detected Face", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # elif confidence > 100:
    #     global camRec
    #     if camRec is False:
    #         camRec = True
    #         Process(target = unknown_user_rec).start()
    
# def registration_time(request):
#     global isRun
#     print(isRun)
#     if isRun:
#         return HttpResponse(True)
#     return HttpResponse(False)

# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
#     faceSamples=[]
#     ids = []
#     detector = cv2.CascadeClassifier("./classifier/haarcascade_frontalface_default.xml")
#     for imagePath in imagePaths:
#         PIL_img = Image.open(imagePath).convert('L')
#         img_numpy = np.array(PIL_img,'uint8')
#         id = int(os.path.split(imagePath)[-1].split("_")[0])
#         faces = detector.detectMultiScale(img_numpy)
#         for (x,y,w,h) in faces:
#             faceSamples.append(img_numpy[y:y+h,x:x+w])
#             ids.append(id)
#     return faceSamples,ids

# def trainer(request):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces,ids = getImagesAndLabels('./data/')
#     recognizer.train(faces, np.array(ids))
#     recognizer.save(f'./trainer/faces.xml')
#     return True

# # ------------------------------
# def home(request):
#     global camera 
#     global which_camera
    
#     if camera != None:
#         camera.release()
#     return render(request, 'monitoring.html', {"which_camera":which_camera})

# def observation_gelen(request):
#     global camera 
#     if camera != None:
#         camera.release()
#     return render(request, 'observation_gelen.html')

# def observation_gelmedik(request):
#     global camera 
#     if camera != None:
#         camera.release()
#     return render(request, 'observation_gelmedik.html')

# def detected_user(request):
#     date = dt.today()
#     year = date.year
#     month = date.month
#     day = date.day
#     if request.method == "GET":
#         users_array = []
#         global which_camera
#         detected_users_id = DetectedUsers.objects.filter(created_at__year = year, created_at__month = month, created_at__day = day).order_by("-id")
#         for i in detected_users_id:
#             user = Profile.objects.get(id = i.user_id)
#             user.time = i.created_at
#             user.job_s = JOBS[f'{user.job}']
#             if dt.strftime(user.time, "%x") == dt.strftime(dt.today(), "%x"):
#                 users_array.append(user)
#         return render(request, 'detected_users.html', {'data': users_array}) 

# def unknown_user(request):
#     if request.method == "GET":
#         users_array = [os.path.join("./static/unknown",f) for f in os.listdir("./static/unknown")][:-6:-1]
#         return render(request, 'unknown_users.html', {'data': users_array}) 

# def unknown_user_analysis(request):
#     if request.method == "GET":
#         users_array = [os.path.join("./static/unknown",f) for f in os.listdir("./static/unknown")][::-1]
#         data=[]
#         for user in users_array:
#             user_time = (user.split("/")[-1][:-4:])
#             data.append(unknownUsers(user, user_time))
#         return render(request, 'unknown_users_analysis.html', {'data': data}) 

# def known_user_analysis(request, value):
#     date = dt.today()
#     year = date.year
#     month = date.month
#     day = date.day
#     if request.method == "GET":
#         data = []
#         detected_users_id = DetectedUsers.objects.filter(created_at__year = year, created_at__month = month, created_at__day = day).order_by("-id")
#         if detected_users_id:
#             for i in detected_users_id:
#                 user = Profile.objects.filter(id = i.user_id, job = value).first()
#                 if user:
#                     user.time = i.created_at
#                     user.job_s = JOBS[f'{user.job}']
#                     user.action_id = i.id
#                     user.action = i.action
#                     data.append(user)
#                     return render(request, 'known_users_analysis.html', {'data': data})
#                 else:
#                     return render(request, 'known_users_analysis.html', {'data': False})
#         else:
#             return render(request, 'known_users_analysis.html', {'data': False})

# def gelmedik_analysis(request, value):
#     date = dt.today()
#     year = date.year
#     month = date.month
#     day = date.day
#     if request.method == "GET":
#         data = []
#         detected_users_id = DetectedUsers.objects.filter(created_at__year = year, created_at__month = month, created_at__day = day).order_by("-id")
#         if detected_users_id:
#             for i in detected_users_id:
#                 data.append(i.user_id)
#             users = Profile.objects.filter(job = value).exclude(id__in = data)
#             data = []
#             if users:
#                 for user in users:
#                     user.time = i.created_at
#                     user.job_s = JOBS[f'{user.job}']
#                     user.action_id = i.id
#                     user.action = i.action
#                     print(user.job_s)
#                     data.append(user)
#                 return render(request, 'gelmedik_analysis.html', {'data': data})
#             else:
#                 return render(request, 'gelmedik_analysis.html', {'data': False})
#         else:
#             users = Profile.objects.filter(job = value)
#             data = []
#             if users:
#                 for user in users:
#                     user.job_s = JOBS[f'{user.job}']
#                     user.action_id = user.id
#                     print(user.job_s)
#                     data.append(user)
#                 return render(request, 'gelmedik_analysis.html', {'data': data})
#             else:
#                 return render(request, 'gelmedik_analysis.html', {'data': False})

#     return render(request, 'gelmedik_analysis.html', {'data': False})

# def unknown_user_delete(request, path):
#     if request.method == "GET":
#         path = path + '.jpg'
#         os.system(f"rm {UNKNOWN_USERS_IMG_PATH}/{path}")
#         return HttpResponse({"data":'True'}) 

# def known_user_delete(request, id):
#     if request.method == "GET":
#         id = int(id)
#         DetectedUsers.objects.get(id=id).delete()
#         return HttpResponse({"data":'True'})

# def registration(request):
#     global camera 
#     if camera != None:
#         camera.release()
#     if request.method == "GET":
#         return render(request, 'register.html', context={"Training":True})
#     elif request.method == "POST":
#         firstname = request.POST["firstname"]
#         surname = request.POST["surname"]
#         fathersname = request.POST["fathersname"]
#         address = request.POST["address"]
#         job = request.POST["job"]
#         phone = request.POST["phone"]
#         email = request.POST["email"]
#         avatar = request.FILES["avatar"]

#         profile = Profile.objects.create(
#             firstname = firstname, 
#             surname = surname, 
#             fathersname = fathersname, 
#             address = address, 
#             job = job, 
#             phone = phone, 
#             email = email,
#             avatar = avatar
#         )
#         if profile:
#             user_id = Profile.objects.filter(firstname = firstname, surname = surname, fathersname = fathersname).first()
#             request.session["face_id"] = f"{user_id.id}_{firstname}_{surname}_{fathersname}"
#         return render(request, 'register.html', {
#             "reg_user":user_id,
#         })

# def get_frame(request, stream_path):
#     global camera
#     camera =cv2.VideoCapture("rtsp://tapocam:neural@213.137.237.165:554/stream1")

#     count, count_frame = 0, 0
#     face_cascade = cv2.CascadeClassifier("./classifier/haarcascade_frontalface_default.xml")
#     if stream_path == "registration":
#         face_id = request.session["face_id"]
#     start = dt.now()
    
#     if camera != None:
#         if camera.isOpened():
#             while True:
#                 count_frame+=1
#                 try:
#                     _, img = camera.read()
#                     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     faces = face_cascade.detectMultiScale(gray, 1.3, 4)

#                     for (x,y,w,h) in faces:
#                         cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 4)

#                         if stream_path == "registration":
#                             count += 1
#                             cv2.imwrite(f"./data/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

#                         if stream_path == "monitoring" and count_frame%16==0:
#                             if faces != ():
#                                 Process(target = recognizer, args=(gray, x, y, w, h, img)).start()

#                     imgencode=cv2.imencode('.jpg',img)[1]
#                     stringData=imgencode.tostring()
#                     yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
#                     if count > 150:
#                         global isRun
#                         isRun = True
#                         trainer(request)
#                         camera.release()
#                 except:
#                     pass
#             del(camera)

# @gzip.gzip_page
# def dynamic_stream(request,stream_path="video"):
#     try :
#         return StreamingHttpResponse(get_frame(request, stream_path),content_type="multipart/x-mixed-replace;boundary=frame")
#     except :
#         return "error"
