import cv2
import numpy as np
from tqdm import tqdm
import sys

# 计算归一化互相关
def cross_correlation(image, template):
    # NCC
    image = image.astype(np.float32)
    template = template.astype(np.float32)

    # 计算均值
    mean_img = np.mean(image)
    mean_tpl = np.mean(template)

    numerator = np.sum((image - mean_img) * (template - mean_tpl))
    denominator = np.sqrt(np.sum((image - mean_img) ** 2) * np.sum((template - mean_tpl) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def process_frame(frame, template_list): 
    global last_x, last_y, cnt
    cnt += 1
    # 备份原始帧用于最终绘制（不经过增强）
    original_frame = frame.copy()  
    
    # 定义一个滑动窗口进行相关性计算
    max_corr = -1  # 初始为一个较低的相关值
    best_match = None
    scale = 0.5
    best_template_height, best_template_width = 0, 0
    list_index = 0 
    best_template = 0

    # 增强当前帧用于匹配（不影响原始帧）
# 使用三个模板
    for template in template_list:
        template_height, template_width = template.shape[:2]  # 获取高和宽，忽略通道数
        list_index += 1

        # 在上一帧的附近进行目标识别与跟踪
        x_min = max(0, last_x - 20)
        x_max = min(frame.shape[1] - template_width, last_x + 20)
        y_min = max(0, last_y - 20)
        y_max = min(frame.shape[0] - template_height, last_y + 20)

        # 滑窗计算最大相关值
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):

                # 提取当前窗口区域
                window = frame[y:y+template_height, x:x+template_width]
                corr = cross_correlation(window, template)
                
                # 更新最大相关度和最佳匹配位置
                if corr > max_corr:
                    max_corr = corr
                    best_match = (x, y)
                    best_template = list_index
                    best_template_height = template_height
                    best_template_width = template_width  
    if best_match:
        x, y = best_match


        # 目标中心
        center_x = int(x + best_template_width / 2)
        center_y = int(y + best_template_height / 2)

        cv2.ellipse(original_frame,
        (center_x, center_y),
        (int(best_template_width/2), int(best_template_height/2)),
        0, 0, 360,
        (0, 255, 0), 1)


        # === 卡尔曼滤波器校正 ===
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measurement)  # 校正：根据当前测量结果更新预测

    else:
        # 如果匹配失败，只用预测（跳过校正）
        measurement = None

    # === 卡尔曼预测 ===
    prediction = kalman.predict()
    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])

    # print("best match:", x,",", y, ", kalman_prediction:", predicted_x,",", predicted_y)
    # === 使用预测结果作为显示中心 ===
    cv2.ellipse(original_frame,
                (predicted_x, predicted_y),
                (int(best_template_width/2), int(best_template_height/2)),
                0, 0, 360,
                (255, 0, 0), 1)
    
    # # 匹配到的位置与预测位置的欧氏距离，如果检测距离太远则使用卡尔曼预测
    # distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
    # print("distance:", distance)
    # if cnt < 90:
    #     last_x, last_y = best_match
    # elif cnt >= 90 and distance < 10:
    #     last_x, last_y = best_match
    # else:
    last_x, last_y = best_match

    trajectory.append((predicted_x, predicted_y))
    for i in range(1, len(trajectory)):
        cv2.line(original_frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 1)

    trajectory2.append((center_x, center_y))
    for i in range(1, len(trajectory2)):
        cv2.line(original_frame, trajectory2[i - 1], trajectory2[i], (0, 255, 255), 1)

    return original_frame

        

# 目标匹配与跟踪
def track_target(video_path, template, output_path="output.mp4"):

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    else:
        print('视频开始处理：', video_path)

    # 获取视频总帧数
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)
    
    
    cap = cv2.VideoCapture(video_path)
    # 获取视频的第一帧
    success, frame = cap.read()
    if not success:
        print("Error: Cannot read video frame.")
        return

    
    # 获取视频帧的尺寸
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

        # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    print("读取错误")
                    break

                # 处理帧
                try:
                    frame = process_frame(frame, template)
                except BaseException as error:
                    print('处理帧报错！', error)
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

    # 释放视频捕捉对象和窗口
    cap.release()
    cv2.destroyAllWindows()

def calculate_stability(points):
    if len(points) < 2:
        return 0.0
    points = np.array(points)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.std(diffs)


if __name__ == "__main__":
    video_path = "大疆无人机航拍视频.mp4"
    last_x = 250
    last_y = 480
    best_template_width, best_template_height = 0, 0
    trajectory = []
    trajectory2 = [] 
    cnt = 0

    template1 = cv2.imread("template1.png")
    template2 = cv2.imread("template2.png")
    template3 = cv2.imread("template3.png")
    template4 = cv2.imread("template4.png")
    template5 = cv2.imread("template5.png")
    template6 = cv2.imread("template6.png")

    template = [template1, template3, template4, template5]

        # 放在主程序初始化阶段
    kalman = cv2.KalmanFilter(4, 2)  # 状态4维(x,y,dx,dy)，测量2维(x,y)
    
    # 状态转移矩阵
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    # 测量矩阵（只观测x,y）
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)

    # # 过程噪声与测量噪声（可以调小以增强平滑）
    # kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    # kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    # 初始化卡尔曼状态为目标初始位置
    kalman.statePre = np.array([[np.float32(last_x)],
                                [np.float32(last_y)],
                                [0],
                                [0]], dtype=np.float32)

    kalman.statePost = kalman.statePre.copy()

    track_target(video_path, template, "output2_kalman.mp4")
    
    # 稳定性计算
    stability = calculate_stability(trajectory2)
    print(f"检测框帧间位置变化的标准差（稳定性指标）：{stability:.2f} 像素")
