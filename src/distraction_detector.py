# 参考 https://github.com/johannesharmse/distraction_detection 的实现
try:
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
    print("✅ 使用 TensorFlow.Keras")
except ImportError:
    try:
        from keras.preprocessing.image import img_to_array
        from keras.models import load_model
        KERAS_AVAILABLE = True
        print("✅ 使用 Keras")
    except ImportError:
        print("❌ Keras/TensorFlow 不可用")
        KERAS_AVAILABLE = False

import cv2
import numpy as np

def resize_frame(frame, width):
    """替代imutils.resize的简单实现"""
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height))

# models - 参考原项目的路径设置
# face and eyes are templates from opencv
try:
    # 尝试使用cv2.data路径（推荐）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("✅ 使用系统级联分类器")
except AttributeError:
    # 如果cv2.data不可用，尝试本地文件
    face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
    print("✅ 使用本地级联分类器")

# 加载训练好的模型 - 参考原项目实现
if KERAS_AVAILABLE:
    try:
        print("正在加载模型...")
        # 参考原项目：distract_model = load_model('cnn/distraction_model.hdf5', compile=False)
        distract_model = load_model('cnn/distraction_model.hdf5', compile=False)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        print("⚠️ 将使用简化检测逻辑")
        distract_model = None
else:
    print("⚠️ Keras不可用，将使用简化检测逻辑")
    distract_model = None

# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

# Video writer
# IMPORTANT:
# - frame width and height must match output frame shape
# - avi works on ubuntu. mp4 doesn't :/
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0,(1200, 900))

# init camera window
cv2.namedWindow('Watcha Looking At?')
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if (camera.isOpened() == False):
    print("Unable to read camera feed")

while True:
    # get frame
    ret, frame = camera.read()

    # if we have a frame, do stuff
    if ret:

        # make frame bigger
        frame = resize_frame(frame, width=frame_w)

        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face(s)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)

        # for each face, detect eyes and distraction
        if len(faces) > 0:
            # loop through faces
            for (x,y,w,h) in faces:
                # draw face rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # get gray face for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                # get colour face for distraction detection (model has 3 input channels - probably redundant)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # 绘制眼睛矩形
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)

                # 使用AI模型或简化逻辑进行检测
                if distract_model is not None and KERAS_AVAILABLE and len(eyes) > 0:
                    # 使用AI模型进行预测
                    probs = list()

                    # loop through detected eyes
                    for (ex,ey,ew,eh) in eyes:
                        # get colour eye for distraction detection
                        roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]

                        if roi.size > 0:
                            # match CNN input shape
                            roi = cv2.resize(roi, (64, 64))
                            # normalize (as done in model training)
                            roi = roi.astype("float") / 255.0
                            # change to array
                            roi = img_to_array(roi)
                            # correct shape
                            roi = np.expand_dims(roi, axis=0)

                            # distraction classification/detection
                            prediction = distract_model.predict(roi, verbose=0)
                            # save eye result
                            probs.append(prediction[0])

                    # get average score for all eyes
                    if probs:
                        probs_mean = np.mean(probs)
                        # get label
                        if probs_mean <= 0.5:
                            label = 'distracted'
                        else:
                            label = 'focused'
                    else:
                        label = 'unknown'
                else:
                    # 使用简化的检测逻辑
                    eye_count = len(eyes)
                    if eye_count >= 2:
                        label = 'focused'
                    elif eye_count == 1:
                        label = 'distracted'
                    else:
                        label = 'distracted'

                # insert label on frame
                cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3, cv2.LINE_AA)

        # Write the frame to video
        video_out.write(frame)

        # display frame in window
        cv2.imshow('Watcha Looking At?', frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # no frame, don't do stuff
    else:
        break

# close
camera.release()
video_out.release()
cv2.destroyAllWindows()

class DistractionDetector:
    """分心检测器类 - 用于API调用"""

    def __init__(self):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.distract_model = distract_model
        self.KERAS_AVAILABLE = KERAS_AVAILABLE

    def detect_frame(self, frame):
        """检测单帧图像中的分心状态"""
        try:
            # 调整帧大小
            frame = resize_frame(frame, width=frame_w)

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbours,
                minSize=(min_size_w, min_size_h),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            results = []

            for (x, y, w, h) in faces:
                # 提取人脸区域
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # 检测眼睛
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbours,
                    minSize=(min_size_w_eye, min_size_h_eye)
                )

                # 使用AI模型或简化逻辑进行检测
                if self.distract_model is not None and self.KERAS_AVAILABLE and len(eyes) > 0:
                    # 使用AI模型进行预测
                    probs = list()

                    for (ex,ey,ew,eh) in eyes:
                        roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]

                        if roi.size > 0:
                            roi = cv2.resize(roi, (64, 64))
                            roi = roi.astype("float") / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)

                            prediction = self.distract_model.predict(roi, verbose=0)
                            probs.append(prediction[0])

                    if probs:
                        probs_mean = np.mean(probs)
                        if probs_mean <= 0.5:
                            label = 'distracted'
                        else:
                            label = 'focused'
                        confidence = float(probs_mean if probs_mean > 0.5 else 1 - probs_mean)
                    else:
                        label = 'unknown'
                        confidence = 0.0
                else:
                    # 使用简化的检测逻辑
                    eye_count = len(eyes)
                    if eye_count >= 2:
                        label = 'focused'
                        confidence = 0.8
                    elif eye_count == 1:
                        label = 'distracted'
                        confidence = 0.6
                    else:
                        label = 'distracted'
                        confidence = 0.7

                results.append({
                    'face_position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'status': label,
                    'confidence': confidence,
                    'eyes_detected': len(eyes)
                })

            return {
                'success': True,
                'faces_detected': len(faces),
                'results': results
            }

        except Exception as e:
            return {
                'success': False,
                'faces_detected': 0,
                'results': [],
                'error': str(e)
            }