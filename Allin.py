# -*- coding: utf-8 -*-
"""
===================================
Created on Fri January  18 09:08:39 2018
@author: Ou
===================================
使用说明：
1：拍照
2：人脸检测
3：人脸识别
q：退出
===================================
"""
import cv2
import face_recognition
import os


def recognizeUs():
    # 这是一个超级简单（但很慢）的例子，在你的网络摄像头上实时运行人脸识别
    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
    # 请注意：这个例子需要安装OpenCV
    # 具体的演示。如果你安装它有困难，试试其他不需要它的演示。
    # 得到一个参考的摄像头# 0（默认）
    video_capture = cv2.VideoCapture(0)
    # 加载示例图片并学习如何识别它。
    path = "./myPic"  # 在同级目录下的images文件中放需要被识别出的人物图
    total_image = []
    total_image_name = []
    total_face_encoding = []
    for fn in os.listdir(path):  # fn 表示的是文件名
        print(fn)
        total_face_encoding.append(
            face_recognition.face_encodings(face_recognition.load_image_file(path + "/" + fn))[0])
        fn = fn[:(len(fn) - 4)]  # 截取图片名（这里应该把images文件中的图片名命名为为人物名）
        total_image_name.append(fn)  # 图片名字列表
    while True:
        # 抓取一帧视频
        ret, frame = video_capture.read()
        # 发现在视频帧所有的脸和face_enqcodings
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        # 在这个视频帧中循环遍历每个人脸
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 看看面部是否与已知人脸相匹配。
            for i, v in enumerate(total_face_encoding):
                match = face_recognition.compare_faces([v], face_encoding, tolerance=0.5)
                name = "Unknown"
                if match[0]:
                    name = total_image_name[i]
                    break
            # 画出一个框，框住脸
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # 画出一个带名字的标签，放在框下
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # 显示结果图像
        cv2.imshow('Video', frame)
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头中的流
    video_capture.release()
    cv2.destroyAllWindows()


def dataOfme():
    camera = cv2.VideoCapture(0)
    str = input("请输入你的名字：")
    while True:
        # get a frame
        ret, frame = camera.read()
        # show a frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("./myPic/" + str + ".jpg", frame)
            break
    cv2.destroyAllWindows()


def detect_rects(img, cascade):
    rects = cascade.detectMultiScale(img, 1.3, 5)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def detect():
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detect_rects(gray, face_cascade)
        frame2 = frame.copy()
        draw_rects(frame2, faces, (255, 0, 0))
        for x1, y1, x2, y2 in faces:
            roi_gray = gray[y1:y2, x1:x2]
            roi = frame2[y1:y2, x1:x2]
            eyes = detect_rects(roi_gray, eye_cascade)
            draw_rects(roi, eyes, (0, 255, 0))
        cv2.imshow('detect', frame2)
        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()


def main():
    while True:
        print(__doc__)
        choice = str(input('请输入数字 >>> '))

        if choice == str(1):
            dataOfme()
        elif choice == str(2):
            detect()
        elif choice == str(3):
            recognizeUs()
        else:
            break


if __name__ == '__main__':
    main()
