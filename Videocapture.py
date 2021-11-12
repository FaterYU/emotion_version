import cv2
import os


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data():
    dirs = os.listdir(r'./img_train')
    faces = []
    labels = []
    for image_path in dirs:
        if image_path[0] == 'h':
            label = 1
        elif image_path[1] == 'a':
            label = 2
        else:
            label = 3
        image_path = './img_train/' + image_path
        image = cv2.imread(image_path, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


def predict(test_img, face_recognizer):
    subjects = ['', 'Happy', 'Sad', 'Surprise']
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        return img
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img
