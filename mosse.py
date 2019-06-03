import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""

class mosse:
    def __init__(self, args, img_path):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
    
    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # get the init ground truth.. [x, y, width, height]
        init_gt = cv2.selectROI('demo', init_img, False, False)    #选取图片中的部分，不用准星，不从随鼠标自动扩展
        init_gt = np.array(init_gt).astype(np.int64)     #获得选取的矩形框的坐标
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]   #得到高斯化之后矩形框中内容
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]    #原图中矩形框内容
        G = np.fft.fft2(g)   #得到理想输出模板在频域中的响应
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)   #第一帧的滤波器get
        # start the tracking...
        time=[]
        for idx in range(len(self.frame_lists)):
            start = cv2.getTickCount()
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                Ai = self.args.lr * Ai    #权值
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi    #得到上一帧滤波器
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]   #获取上一帧锁定位置
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                Gi = Hi * np.fft.fft2(fi)   #理想输出模板获取
                gi = linear_mapping(np.fft.ifft2(Gi))  #转到空域
                # find the max pos...
                max_value = np.max(gi)  #找到相应最大的值
                max_pos = np.where(gi == max_value)  #响应最大的坐标
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                
                # update the position...
                pos[0] = pos[0] + dx   #确定新的锁定框的x坐标
                pos[1] = pos[1] + dy   #确定新的锁定框的y坐标

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])   #控制矩形框不要超出图片范围
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))   #根据当前选中的框大小调整滤波器的大小
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi   #加权下一帧滤波器
            
            # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)  #圈出锁定的目标
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.jpg', current_frame)
            end = cv2.getTickCount()
            time.append((end - start) / cv2.getTickFrequency()*1000)
        print(time)
        print(np.mean(time))





    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):    #应该是在输出最佳滤波器
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))     #先对fi快速傅立叶变换后取共轭，再相乘
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))     #获得height行x可能的取值，width列y可能的取值
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]    #获取矩形框中心x坐标
        center_y = gt[1] + 0.5 * gt[3]    #获得矩形框中心y坐标
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)    #记录图片上每个点坐标到中心距离平方从除以2sigama
        # get the response map...
        response = np.exp(-dist)  #e^-dist
        # normalize...
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    def _get_img_lists(self, img_path):      #获取每一帧的图片
        frame_list = []
        for frame in os.listdir(img_path):  #读取文件夹中的文件
            if os.path.splitext(frame)[1] == '.jpg':  #如果后缀名为jpg
                frame_list.append(os.path.join(img_path, frame))    #则将其路径加入frame_list中
        return frame_list
    
    # it will get the first ground truth of the video..
    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]

