# from prepare_training_data import prepare_training_data
# import cv2
# import numpy as np
# from predict import predict
import cv2
import numpy as np
from Videocapture import *

if __name__ == '__main__':
    print("Preparing data...")
    #调用之前写的函数，得到包含多个人脸矩阵的序列和它们对于的标签
    faces, labels = prepare_training_data()
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    #得到（LBPH）人脸识别器
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #应用数据，进行训练
    face_recognizer.train(faces, np.array(labels))
    print("Predicting images...")

    screen = cv2.VideoCapture(0)

    while 1:
        success, img = screen.read()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if success == 0:
            continue
        predict_img = predict(img, face_recognizer)
        cv2.imshow("now_img", predict_img)
        k = cv2.waitKey(1)
        if k == 27:
            # 通过esc键退出摄像
            cv2.destroyAllWindows()
            break
    screen.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 加载预测图像，这里我图简单，就直接把路径写上去了
    test_img1 = cv2.imread(r"./img_predict/happy1.jpg",0)
    sc = cv2.VideoCapture(0)
    tu, te = sc.read()
    # test_img2 = cv2.imread(r"./img_predict/sad1.jpg",0)
    # test_img3 = cv2.imread(r"./img_predict/surprise1.jpg", 0)
    # test_img4 = cv2.imread(r"./img_predict/surprise2.jpg", 0)


    # print(te)
    # print(test_img1)
    # gray1 = cv2.cvtColor(te, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(test_img1, cv2.COLOR_BAYER_BG2BGR)
    # gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
    # print(gray1)
    # print(gray2)
    # cv2.imshow('gray1', gray1)
    # cv2.imshow('gray2', gray2)
    # cv2.waitKey(0)


    # 进行预测
    # predicted_img1 = predict(test_img1, face_recognizer)
    # predicted_img2 = predict(test_img2, face_recognizer)
    # predicted_img3 = predict(test_img3, face_recognizer)
    # predicted_img4 = predict(test_img4, face_recognizer)
    # print("Prediction complete")

    # 显示预测结果
    # cv2.imshow('Happy', predicted_img1)
    # cv2.imshow('Sad', predicted_img2)
    # cv2.imshow('Surprise1', predicted_img3)
    # cv2.imshow('Surprise2', predicted_img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
