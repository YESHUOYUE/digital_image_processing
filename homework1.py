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
def ssd_match(image_patch, template):
    """
    基于SSD的图像匹配函数：计算图像块与模板之间的差的平方和
    :param image_patch: 与模c板同尺寸的图像区域（H, W, 3）
    :param template: 模板图像（H, W, 3）
    :return: SSD 值（越小越相似）
    """
    # 转为 float 避免溢出
    image_patch = image_patch.astype(np.float32)
    template = template.astype(np.float32)
    
    # 计算平方差
    diff = image_patch - template
    ssd = np.sum(diff ** 2)
    
    return ssd

def process_frame(frame, template_list):
    global last_x, last_y
    # 定义一个滑动窗口进行相关性计算
    max_corr = -1  # 初始为一个较低的相关值
    min_ssd = sys.maxsize
    best_match = None
    scale = 0.5 # 初始化缩小倍数
    best_template_height, best_template_width = 0, 0
    list_index = 0 # 正在使用的模板index
    best_template = 0

    # 使用三个模板
    for template in template_list:
        template_height, template_width = template.shape[:2]  # 获取高和宽，忽略通道数
        list_index += 1
        # 优化策略
        if last_x == None:
            # 第一帧加载速度过慢，将第一帧进行压缩后进行大概定位，速度有显著提升 
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            x_min = 0
            x_max = frame.shape[1] - template_width
            y_min = 0
            y_max = frame.shape[0] - template_height
            scale = 0.5
            template_height, template_width = template.shape[:2]  # 获取高和宽，忽略通道数
        else:
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
                
                # 计算当前窗口与模板的相关值
                corr = ssd_match(window, template)
                # print("ssd:", ssd)
                # corr = cross_correlation(window, template)
                
                # 更新最大相关度和最佳匹配位置
                if corr > max_corr:
                # if ssd < min_ssd:
                    # min_ssd = ssd
                    max_corr = corr
                    best_match = (x, y)
                    best_template = list_index
                    best_template_height = template_height
                    best_template_width = template_width
        
    # 如果找到最佳匹配，绘制矩形框标出目标位置
    if best_match:
        # 第一帧压缩后恢复
        if last_x is None:
            best_match = (int(best_match[0] / scale), int(best_match[1] / scale))
        # print("score:", ssd, ", best_template:", best_template, ", (x, y):", best_match)
        x, y = best_match
        last_x, last_y = best_match

        # 绘制目标的矩形框
        cv2.rectangle(frame, (x, y), (x + best_template_width, y + best_template_height), (0, 255, 0), 2) 

        # 绘制运动轨迹(人物中心像素点)
        center_x = int(x + best_template_width / 2)
        center_y = int(y + best_template_height / 2)
        trajectory.append((center_x, center_y))
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        # cv2.circle(frame, (int(x + template_width/2), int(y + template_height/2)),15,(0,0,255),-1)  # 绘制人物中心像素点
    
    return frame
        

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

    # 加载视频文件和目标模板
    video_path = "动画表情视频.mp4"
    template1 = cv2.imread("small_template_1.png")  # 读取RGB模板
    template2 = cv2.imread("small_template_2.png") 
    template3 = cv2.imread("small_template_3.png") 
    template = [template1, template2, template3]

    last_x = None
    last_y = None
    trajectory = []
    FNR = 0

    # 开始目标跟踪
    track_target(video_path, template, "output1_stability.mp4")

        # 稳定性计算
    stability = calculate_stability(trajectory)
    print(f"检测框帧间位置变化的标准差（稳定性指标）：{stability:.2f} 像素")
