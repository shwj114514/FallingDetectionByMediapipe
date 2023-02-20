import cv2
import mediapipe as mp
import numpy as np
import torch

from train import shwj

mp_pose = mp.solutions.pose

# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=True,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                    )


# 对tensor转numpy求最大值
def softmax(x):
    x = x[0].numpy()
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    print("摔倒概率为:", y[1])


def predict(img,model):
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    pos = []
    for i in range(33):
        # 获取该关键点的三维坐标
        cx = results.pose_landmarks.landmark[i].x
        cy = results.pose_landmarks.landmark[i].y
        cz = results.pose_landmarks.landmark[i].z
        pos.append(cx)
        pos.append(cy)
        pos.append(cz)
    pos_input = torch.tensor(pos)
    pos_input = pos_input.unsqueeze(0)
    output_result = model(pos_input)
    output_result = output_result.detach()
    softmax(output_result)

if __name__ == '__main__':
    shwj = torch.load("fall_vs_up.pth")
    img = cv2.imread('fall3.jpg')
    predict(img,model=shwj)
