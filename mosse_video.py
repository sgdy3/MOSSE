import cv2
import os
img_path=r'E:\python trial\python now\mosse\record_frames\surfer'
frame_list = []
for frame in os.listdir(img_path):  # 读取文件夹中的文件
    if os.path.splitext(frame)[1] == '.jpg':  # 如果后缀名为jpg
        frame_list.append(os.path.join(img_path, frame))  # 则将其路径加入frame_list中
fps=20
img=cv2.imread(frame_list[0])
size = (int(img.shape[0]),int(img.shape[1]))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 写入fourcc
out = cv2.VideoWriter('cvcourse.avi',fourcc,fps,(size[1],size[0]))
for item in frame_list:
    img=cv2.imread(item)
    out.write(img)
out.release()
print("down")

