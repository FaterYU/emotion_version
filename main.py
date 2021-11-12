from prepare_training_data import prepare_training_data
import cv2
import numpy as np
from predict import predict

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
        predict_img = predict(img, face_recognizer)
        cv2.imshow("now_img", predict_img)
        k = cv2.waitKey(0)
        if k == 27:
            # 通过esc键退出摄像
            cv2.destroyAllWindows()
            break
        elif k == ord("s"):
            # 通过s键保存图片，并退出。
            cv2.imwrite("image2.jpg", img)
            cv2.destroyAllWindows()
            break
        cv2.waitKey(0)
        break
    screen.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 加载预测图像，这里我图简单，就直接把路径写上去了
    # test_img1 = cv2.imread(r"./img_predict/happy1.jpg",0)
    # test_img2 = cv2.imread(r"./img_predict/sad1.jpg",0)
    # test_img3 = cv2.imread(r"./img_predict/surprise1.jpg", 0)
    # test_img4 = cv2.imread(r"./img_predict/surprise2.jpg", 0)

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
