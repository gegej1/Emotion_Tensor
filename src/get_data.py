from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
import numpy as np
import os
import random

# models
# face and eyes are templates from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

# image iterators
# i = image filename number
# j = controls how often images should be saved
i = 0
j = 0

# 创建目录（如果不存在）
train_focused_dir = '../data/train/focused'
train_distracted_dir = '../data/train/distracted'
validate_focused_dir = '../data/validate/focused'
validate_distracted_dir = '../data/validate/distracted'

for directory in [train_focused_dir, train_distracted_dir, validate_focused_dir, validate_distracted_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 当前模式：0=专注，1=分心
current_mode = 0
# 当前保存位置：0=训练集，1=验证集
save_to_validation = False
# 验证集比例
validation_ratio = 0.2

# 状态显示文本 - 使用英文避免乱码
mode_text = "FOCUSED MODE (Press 'd' to switch)"
save_text = "SAVING TO TRAINING SET (Press 'v' to switch)"

# init camera window
cv2.namedWindow('Watcha Looking At?')
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if (camera.isOpened() == False): 
    print("Unable to read camera feed")

print("Key instructions:")
print("d - Switch mode (Focused/Distracted)")
print("v - Switch save location (Training/Validation)")
print("s - Save currently detected eyes")
print("q - Quit program")

while True:
    # get frame
    ret, frame = camera.read()

    # if we have a frame, do stuff
    if ret:
        
        # make frame bigger
        frame = imutils.resize(frame,width=frame_w)

        # 显示当前模式和保存位置
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, save_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face(s)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)

        detected_eyes = []  # 存储检测到的眼睛图像

        # for each face, detect eyes and distraction
        if len(faces) > 0:
            # loop through faces
            for (x,y,w,h) in faces:
                # draw face rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # get gray face for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                # get colour face for saving colour eye images for CNN (probs not necessary)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # 提取眼睛图像
                    eye_img = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                    detected_eyes.append(eye_img)

        # show frame in window
        cv2.imshow('Watcha Looking At?', frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # 退出
            break
        elif key == ord('d'):  # 切换模式
            current_mode = 1 - current_mode
            mode_text = "DISTRACTED MODE (Press 'd' to switch)" if current_mode == 1 else "FOCUSED MODE (Press 'd' to switch)"
        elif key == ord('v'):  # 切换保存位置
            save_to_validation = not save_to_validation
            save_text = "SAVING TO VALIDATION SET (Press 'v' to switch)" if save_to_validation else "SAVING TO TRAINING SET (Press 'v' to switch)"
        elif key == ord('s'):  # 手动保存
            if detected_eyes:
                for eye_img in detected_eyes:
                    # 创建新文件名
                    i += 1
                    
                    # 确定保存目录
                    if save_to_validation:
                        save_dir = validate_distracted_dir if current_mode == 1 else validate_focused_dir
                    else:
                        save_dir = train_distracted_dir if current_mode == 1 else train_focused_dir
                    
                    # 指定保存位置
                    filename = os.path.join(save_dir, f'eye_{i}.jpg')
                    
                    # 调整大小为64x64
                    eye_img_resized = cv2.resize(eye_img, (64, 64))
                    
                    # 保存图像
                    cv2.imwrite(filename, eye_img_resized)
                    print(f"Saved: {filename}")
                
                print(f"Saved {len(detected_eyes)} eye images")

# close
camera.release()
cv2.destroyAllWindows()
