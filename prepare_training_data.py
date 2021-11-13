import os

from detect_face import detect_face
import cv2


def prepare_training_data():
    #读取训练文件夹中的图片名称
    dirs = os.listdir(r'./img_train')
    faces = []
    labels = []
    for image_path in dirs:
        #如果图片的名称以happy开头，则标签为1l；sad开头，标签为2
        if image_path[0] == 'h':
            label = 1
        elif image_path[1] == 'a':
            label = 2
        else:
            label = 3

        #得到图片路径
        image_path = './img_train/' + image_path

        #返回灰度图，返回Mat对象
        image = cv2.imread(image_path, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        #以窗口形式显示图像，显示100毫秒
        # cv2.imshow("Training on image...", image)
        # cv2.waitKey(50)


        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
