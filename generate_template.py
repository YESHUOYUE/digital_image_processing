import cv2
if __name__ == "__main__":

    # 任务2
    template = cv2.imread("homework2_target.png")  # 读取RGB模板
    x1, y1, w1, h1 = 480, 450, 80, 60  
    template1 = template[y1:y1+h1, x1:x1+w1]
    cv2.imwrite("template1.png", template1)

    template = cv2.imread("homework2_target2.png")
    x2, y2, w2, h2 = 225, 305, 50, 25
    template2 = template[y2:y2+h2, x2:x2+w2]
    cv2.imwrite("template2.png", template2)

    template = cv2.imread("homework2_target3.png")
    x3, y3, w3, h3 = 200, 130, 20, 15
    template3 = template[y3:y3+h3, x3:x3+w3]
    cv2.imwrite("template3.png", template3)

    template = cv2.imread("homework2_target3.png")
    x3, y3, w3, h3 = 195, 125, 30, 25
    template6 = template[y3:y3+h3, x3:x3+w3]
    cv2.imwrite("template6.png", template6)


    template = cv2.imread("homework2_target4.png")
    x4, y4, w4, h4 = 200, 255, 45, 45
    template4 = template[y4:y4+h4, x4:x4+w4]
    cv2.imwrite("template4.png", template4)
    
    template = cv2.imread("homework2_target5.png")
    x5, y5, w5, h5 = 210, 145, 28, 28
    template5 = template[y5:y5+h5, x5:x5+w5]
    cv2.imwrite("template5.png", template5)

    #任务3
    template = cv2.imread("homework3_target.png")  # 读取RGB模板
    x1, y1, w1, h1 = 343, 147, 7, 10  
    t1 = template[y1:y1+h1, x1:x1+w1]
    cv2.imwrite("t1.png", t1)

    template = cv2.imread("homework3_target2.png")  # 读取RGB模板
    x2, y2, w2, h2 = 725, 430, 25, 28  
    t2 = template[y2:y2+h2, x2:x2+w2]
    cv2.imwrite("t2.png", t2)

