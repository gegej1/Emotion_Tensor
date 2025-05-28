"""
分心检测模块 - 基于OpenCV和TensorFlow/Keras
提供人脸检测、眼睛检测和分心状态分析功能
"""

import os
import cv2
import numpy as np

# 尝试导入TensorFlow/Keras
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

# 常量定义
FRAME_WIDTH = 1200
BORDER_WIDTH = 2
MIN_FACE_WIDTH = 240
MIN_FACE_HEIGHT = 240
MIN_EYE_WIDTH = 60
MIN_EYE_HEIGHT = 60
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

def resize_frame(frame, width):
    """调整帧大小的辅助函数"""
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height))

def load_cascade_classifiers():
    """加载OpenCV级联分类器"""
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

    return face_cascade, eye_cascade

def load_distraction_model():
    """加载分心检测模型"""
    if not KERAS_AVAILABLE:
        print("⚠️ Keras不可用，将使用简化检测逻辑")
        return None

    try:
        print("正在加载模型...")
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)

        # 尝试多个可能的路径
        model_paths = [
            # 相对于当前文件的路径
            os.path.join(current_dir, 'cnn', 'distraction_model.hdf5'),
            # 相对于当前工作目录的路径
            'distraction_detection/src/cnn/distraction_model.hdf5',
            # 其他可能的路径
            'src/cnn/distraction_model.hdf5',
            'cnn/distraction_model.hdf5',
            '../cnn/distraction_model.hdf5'
        ]

        for path in model_paths:
            try:
                print(f"尝试加载模型: {path}")
                if os.path.exists(path):
                    model = load_model(path, compile=False)
                    print(f"✅ 模型从 {path} 加载成功！")
                    return model
                else:
                    print(f"  - 路径 {path} 不存在")
            except Exception as e:
                print(f"  - 路径 {path} 加载失败: {e}")

        print("❌ 加载模型时出错: 所有路径都加载失败")
        return None
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        print("⚠️ 将使用简化检测逻辑")
        return None

class DistractionDetector:
    """分心检测器类 - 用于API调用"""

    def __init__(self):
        """初始化分心检测器"""
        self.face_cascade, self.eye_cascade = load_cascade_classifiers()
        self.distract_model = load_distraction_model()
        self.KERAS_AVAILABLE = KERAS_AVAILABLE

    def detect_frame(self, frame):
        """检测单帧图像中的分心状态

        参数:
            frame: OpenCV格式的图像帧

        返回:
            包含检测结果的字典
        """
        try:
            # 调整帧大小
            frame = resize_frame(frame, width=FRAME_WIDTH)

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT),
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
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=(MIN_EYE_WIDTH, MIN_EYE_HEIGHT)
                )

                # 分析分心状态
                status, confidence = self._analyze_distraction(roi_color, eyes)

                results.append({
                    'face_position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'status': status,
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

    def _analyze_distraction(self, roi_color, eyes):
        """分析分心状态

        参数:
            roi_color: 人脸区域的彩色图像
            eyes: 检测到的眼睛区域

        返回:
            (status, confidence): 状态和置信度
        """
        # 使用AI模型或简化逻辑进行检测
        if self.distract_model is not None and self.KERAS_AVAILABLE and len(eyes) > 0:
            # 使用AI模型进行预测
            probs = []

            for (ex, ey, ew, eh) in eyes:
                roi = roi_color[ey+BORDER_WIDTH:ey+eh-BORDER_WIDTH, ex+BORDER_WIDTH:ex+ew-BORDER_WIDTH]

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
                    status = 'distracted'
                else:
                    status = 'focused'
                confidence = float(probs_mean if probs_mean > 0.5 else 1 - probs_mean)
            else:
                status = 'unknown'
                confidence = 0.0
        else:
            # 使用简化的检测逻辑
            eye_count = len(eyes)
            if eye_count >= 2:
                status = 'focused'
                confidence = 0.8
            elif eye_count == 1:
                status = 'distracted'
                confidence = 0.6
            else:
                status = 'distracted'
                confidence = 0.7

        return status, confidence


# 独立运行时的演示代码
if __name__ == "__main__":
    # 初始化视频捕获
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头")
        exit()

    # 初始化检测器
    detector = DistractionDetector()

    # 初始化视频输出
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0, (FRAME_WIDTH, 900))

    # 创建窗口
    cv2.namedWindow('分心检测演示')

    print("按 'q' 键退出")

    while True:
        # 获取帧
        ret, frame = camera.read()
        if not ret:
            break

        # 调整帧大小
        frame = resize_frame(frame, width=FRAME_WIDTH)

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector.face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 处理每个人脸
        for (x, y, w, h) in faces:
            # 绘制人脸矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 提取人脸区域
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # 检测眼睛
            eyes = detector.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(MIN_EYE_WIDTH, MIN_EYE_HEIGHT)
            )

            # 绘制眼睛矩形
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), BORDER_WIDTH)

            # 分析分心状态
            status, confidence = detector._analyze_distraction(roi_color, eyes)

            # 在帧上显示标签
            cv2.putText(
                frame,
                f"{status} ({confidence:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        # 显示帧
        cv2.imshow('分心检测演示', frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    camera.release()
    video_out.release()
    cv2.destroyAllWindows()
