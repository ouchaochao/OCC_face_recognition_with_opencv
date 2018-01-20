# -*- coding: utf-8 -*-

import cv2


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


if __name__ == '__main__':
    detect()