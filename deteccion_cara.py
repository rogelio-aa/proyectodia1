import cv2  

# Cargar los clasificadores en cascada
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if faceCascade.empty() or eyeCascade.empty():
    exit()

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Detección de bordes
    edges = cv2.Canny(gray, 100, 200)

    # Mostrar los bordes en una ventana separada
    cv2.imshow('Detección de Bordes', edges)

    # Mostrar la imagen original con detección de rostros y ojos
    cv2.imshow('Detección de Rostros y Ojos', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()