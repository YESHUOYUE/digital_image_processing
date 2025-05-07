# import cv2

# coords = []

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"你点击了坐标：({x}, {y})")
#         coords.append((x, y))
#         cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         cv2.imshow("image", img)

# img = cv2.imread("homework3_target2.png")  # 确保路径正确
# cv2.imshow("image", img)
# cv2.setMouseCallback("image", click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# # 初始化变量
# x, y, w, h = 0, 0, 0, 0  # 初始化坐标
# drawing = False  # 标记是否正在绘制矩形框

# # 鼠标点击回调函数
# def click_and_drag(event, _x, _y, flags, param):
#     global x, y, w, h, drawing
#     if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
#         drawing = True
#         x, y = _x, _y  # 记录起始点
#     elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动
#         if drawing:
#             w, h = _x - x, _y - y  # 计算矩形框的宽高
#             img_copy = frame.copy()
#             cv2.rectangle(img_copy, (x, y), (_x, _y), (0, 255, 0), 2)
#             cv2.imshow("Frame", img_copy)
#     elif event == cv2.EVENT_LBUTTONUP:  # 左键松开
#         drawing = False
#         w, h = _x - x, _y - y  # 记录矩形框的宽高
#         cv2.rectangle(frame, (x, y), (_x, _y), (0, 255, 0), 2)  # 绘制矩形框
#         cv2.imshow("Frame", frame)
#         print(f"目标位置：x={x}, y={y}, w={w}, h={h}")

# # 读取视频
# video_path = "大疆无人机航拍骑车人.mp4"
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()  # 读取第一帧
# cap.release()

# if not ret:
#     print("无法读取视频的第一帧")
# else:
#     cv2.imshow("Frame", frame)
#     cv2.setMouseCallback("Frame", click_and_drag)
#     cv2.waitKey(0)  # 等待按键关闭窗口
#     cv2.destroyAllWindows()

import cv2

# 初始化变量
x1, y1 = -1, -1  # 鼠标按下的起始点
drawing = False  # 标记是否正在绘制矩形框

# 鼠标回调函数，用于绘制矩形框并记录坐标
def click_and_drag(event, x, y, flags, param):
    global x1, y1, drawing, frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
        drawing = True
        x1, y1 = x, y  # 记录起始点
    elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键松开
        drawing = False
        cv2.rectangle(frame, (x1, y1), (x, y), (0, 255, 0), 2)  # 绘制矩形框
        print(f"矩形框的坐标: x1={x1}, y1={y1}, x2={x}, y2={y}")
        cv2.imshow("Frame", frame)

# 打开视频
cap = cv2.VideoCapture("大疆无人机航拍视频.mp4")
# 获取视频的总帧数和帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 指定帧索引
frame_index = 90  # 例如获取第 100 帧

# 读取视频某一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 设置视频读取的位置
success, frame = cap.read()

if success:

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_and_drag)  # 设置鼠标回调函数
    cv2.waitKey(0)  # 等待按键后关闭窗口
    cv2.destroyAllWindows()

else:
    print("无法读取该帧")

cap.release()

