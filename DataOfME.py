import cv2

camera = cv2.VideoCapture(0)
str = input("请输入你的名字：")
while True:
    # get a frame
    ret, frame = camera.read()
    # show a frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("./myPic/" + str + ".jpg", frame)
        break
cv2.destroyAllWindows()
