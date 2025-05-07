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
    
    # 直接进行np的矩阵运算
    # return np.sum(image * template)
    # return result


def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用CLAHE（局部直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def process_frame(frame, template_list): 
    global last_x, last_y

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
    enhanced_frame = enhance_contrast(frame)

    for template in template_list:
        # 增强模板图像
        template = enhance_contrast(template)
        template_height, template_width = template.shape[:2]
        list_index += 1

        if last_x is None:
            x_min = 0
            x_max = enhanced_frame.shape[1] - template_width
            y_min = 0
            y_max = enhanced_frame.shape[0] - template_height
        else:
            x_min = max(0, last_x - 40)
            x_max = min(enhanced_frame.shape[1] - template_width, last_x + 40)
            y_min = max(0, last_y - 40)
            y_max = min(enhanced_frame.shape[0] - template_height, last_y + 40)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                window = enhanced_frame[y:y+template_height, x:x+template_width]
                corr = cross_correlation(window, template)
                if corr > max_corr:
                    max_corr = corr
                    best_match = (x, y)
                    best_template = list_index
                    best_template_height = template_height
                    best_template_width = template_width
    
    # if max_corr < 0.3:  # 相关性低，目标可能丢失
    #     # 启用全图搜索（重新定位）
    #     x_min, x_max = 0, enhanced_frame.shape[1] - template_width
    #     y_min, y_max = 0, enhanced_frame.shape[0] - template_height
    #     last_x, last_y = None  # 清空滑窗中心
        
# 最终在原图 original_frame 上绘图
    if best_match:
        x, y = best_match
        last_x, last_y = best_match

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

    # === 使用预测结果作为显示中心 ===
    cv2.ellipse(original_frame,
                (predicted_x, predicted_y),
                (int(best_template_width/2), int(best_template_height/2)),
                0, 0, 360,
                (0, 255, 255), 1)

    trajectory.append((predicted_x, predicted_y))
    for i in range(1, len(trajectory)):
        cv2.line(original_frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 1)

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


if __name__ == "__main__":

    # 加载视频文件和目标模板
    video_path = "大疆无人机航拍骑车人.mp4"
    
    last_x = 255
    last_y = 180
    trajectory = []

    # template1 = cv2.imread("t1.png")  # 读取RGB模板
    template2 = cv2.imread("t2.png")
    # template3 = cv2.imread("template3.png")
    # template4 = cv2.imread("template4.png")
    # template5 = cv2.imread("template5.png")


    template = [ template2]

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

    # 过程噪声与测量噪声（可以调小以增强平滑）
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    # 初始化卡尔曼状态为目标初始位置
    kalman.statePre = np.array([[np.float32(last_x)],
                                [np.float32(last_y)],
                                [0],
                                [0]], dtype=np.float32)

    kalman.statePost = kalman.statePre.copy()


    # 开始目标跟踪
    track_target(video_path, template, "output3_2.mp4")

