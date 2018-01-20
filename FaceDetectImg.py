import cv2
import glob


def detect(img, face_patterns):
    rects = face_patterns.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


face_patterns = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_patterns = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
images = glob.glob('img/people*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detect(gray, face_patterns)
    img2 = img.copy()
    draw_rects(img2, rects, (0, 255, 0))
    if not eye_patterns.empty():
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            img2_roi = img2[y1:y2, x1:x2]
            subrects = detect(roi.copy(), eye_patterns)
            draw_rects(img2_roi, subrects, (255, 0, 0))
    cv2.imshow('face', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
