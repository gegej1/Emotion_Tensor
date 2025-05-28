#!/usr/bin/env python3
"""
AI检测API服务 - 连接真正的distraction_detector.py
为学生端提供HTTP API接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image
import uvicorn
import sys
import os

# 添加src目录到路径，以便导入distraction_detector
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# 导入真正的检测器类
from distraction_detector import DistractionDetector

app = FastAPI(
    title="AI Emotion Detection API",
    description="基于真正的distraction_detector.py的AI情感检测服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector = None

class ImageData(BaseModel):
    image: str  # base64编码的图像

class DetectionResult(BaseModel):
    success: bool
    faces_detected: int
    results: list
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化AI检测器"""
    global detector
    try:
        print("🚀 正在初始化真正的AI检测器...")
        print(f"📁 当前工作目录: {os.getcwd()}")
        print(f"📁 src目录路径: {src_dir}")

        # 切换到src目录以确保模型路径正确
        os.chdir(src_dir)
        print(f"📁 切换到目录: {os.getcwd()}")

        # 初始化检测器
        detector = DistractionDetector()
        print("✅ 真正的AI检测器初始化成功！")

    except Exception as e:
        print(f"❌ AI检测器初始化失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        detector = None

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Emotion Detection API",
        "status": "running",
        "detector_loaded": detector is not None
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "message": "AI Emotion Detection API is running",
        "detector_loaded": detector is not None,
        "working_directory": os.getcwd()
    }

@app.post("/analyze-base64", response_model=DetectionResult)
async def analyze_base64_image(data: ImageData):
    """分析base64编码的图像 - 使用真正的AI检测器"""
    if detector is None:
        raise HTTPException(status_code=500, detail="AI检测器未初始化")

    try:
        print(f"📥 收到图像分析请求，数据大小: {len(data.image)} 字符")

        # 解码base64图像
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data))

        # 转换为OpenCV格式 (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        print(f"🖼️ 图像尺寸: {image_cv.shape}")

        # 使用真正的AI检测器分析图像
        result = detector.detect_frame(image_cv)

        print(f"🎯 AI检测结果: {result}")

        return DetectionResult(
            success=result['success'],
            faces_detected=result['faces_detected'],
            results=result['results'],
            error=result.get('error')
        )

    except Exception as e:
        error_msg = f"图像处理错误: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/test")
async def test_detector():
    """测试检测器状态"""
    if detector is None:
        return {"status": "error", "message": "检测器未初始化"}

    return {
        "status": "ok",
        "message": "检测器已就绪",
        "detector_type": type(detector).__name__,
        "working_directory": os.getcwd()
    }

if __name__ == "__main__":
    print("🚀 启动AI检测API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
