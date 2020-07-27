import cv2

print("test start")
camera = cv2.VideoCapture(1)
print("test open")
camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

print("test1")

while True:
    ret, image = camera.read()
    print("test2")
    # 这一步根据实际需要来
    # 由于摄像头是定焦摄像头
    # 而模型的接口是[228,228]
    # 所以这一步就是让读取到的图片也是[A,A]型
    image = image[0:780,150:1050] 
    cv2.imshow('Webcam', image)
    print("test3")
    if cv2.waitKey(0):
        pass

print("test3")
camera.release()
cv2.destroyAllWindows()
