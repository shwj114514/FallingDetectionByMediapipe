import time

import mediapipe as mp
import torch
from predict_img import softmax
from train import shwj
import cv2
from tqdm import tqdm


mp_pose = mp.solutions.pose

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

model=torch.load("fall_vs_up.pth")

def predict(key_point_list):
    input_tensor=torch.tensor(key_point_list)
    # (input_ceshi).shape
    input_tensor=input_tensor.unsqueeze(0)
    output_tensor=shwj(input_tensor)
    output_tensor=output_tensor.detach()
    return softmax(output_tensor)



# 处理帧函数
def process_frame(img):
    text = ""
    # 记录该帧开始处理的时间
    start_time = time.time()

    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks:  # 若检测出人体关键点

        pos = []

        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):  # 遍历所有33个关键点，可视化

            # 获取该关键点的三维坐标
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y
            z = results.pose_landmarks.landmark[i].z
            # 图像中的坐标
            cx = int(x * w)
            cy = int(y * h)
            cz = z

            pos.append(x)
            pos.append(y)
            pos.append(z)

            radius = 10

            if i == 0:  # 鼻尖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11, 12]:  # 肩膀
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:  # 髋关节
                img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:  # 胳膊肘
                img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:  # 膝盖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:  # 手腕和脚腕
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)

            #             elif i in [17,19,21]: # 左手
            #                 img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            #             elif i in [18,20,22]: # 右手
            #                 img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)

            elif i == 27:  # 左脚
                img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
            elif i == 28:  # 右脚
                img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [7, 8]:  # 眼及脸颊
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            else:  # 其它关键点
                #                 img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)
                pass
        posibiliy = predict(pos)
        #         if posibiliy>0.5:
        text = str(posibiliy)

    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)
        # print('从图像中未检测出人体关键点，报错。')
        text = "safe"

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    #     img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)) + '       fall_probability      ' + text, (25 * scaler, 50 * scaler),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    return img


def generate_video(input_path='./videos/output.mp4'):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                # 处理帧
                # frame_path = './temp_frame.png'
                # cv2.imwrite(frame_path, frame)
                try:
                    frame = process_frame(frame)
                except:
                    print('error')
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

if __name__ == '__main__':
    generate_video(input_path='ceshi.mp4')