import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0) #参数选择0即指向第一个摄像头
img_path="mosse/shoot/"
idx=0
while(True):
    # 截取帧，ret为bool型，表征是否截取成功，frame为截取到的图片
    ret, frame = cap.read()
    if(ret):
        # 显示截取的帧
        cv2.imshow('frame', frame)
        frame_path = 'mosse' +'/'+ img_path.split('/')[1] + '/'
        cv2.imwrite(frame_path + str(idx).zfill(5) + '.jpg', frame)
        idx+=1
    else:
        print("调用摄像头失败")
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):  #waitKey(1) 中的数字代表等待按键输入之前的无效时间，单位为毫秒，在这个时间段内按键 ‘q’ 不会被记录
        break