import dlib
import cv2

# 加载并初始化检测器
detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(r'E:\emotion_recognition\videos\201811_V3_5.mp4')
if not camera.isOpened():
    print("cannot open camear")
    exit(0)
j=0
while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 检测脸部
    dets = detector(frame_new, 1)
    print("Number of faces detected: {}".format(len(dets)))
    # 查找脸部位置
    for i, face in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
            i, face.left(), face.top(), face.right(), face.bottom()))
        # 绘制脸部位置
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
        #保存脸部图片
        img1=frame[face.top():face.bottom(),face.left():face.right()]
        cv2.imwrite(r"E:\emotion_recognition\pictures\cutface\face"+str(j)+'.jpg',img1)
        j=j+1
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

