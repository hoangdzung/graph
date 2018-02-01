import numpy as np 
import os
import cv2

class video_Obj:
    def __init__(self, path, optical_flow_channels = 10, size = (256, 256), label = None):
        self.path = path
        self.optical_flow_channels = optical_flow_channels
        self.size = size
        self.label = label
        self._create_temporal_spatial()

    def _create_temporal_spatial(self):
        count_frame = 0
        cap = cv2.VideoCapture(self.path)
        # self.path = self.path.replace('/media/trungdunghoang/4022D29E22D297EC/data', '..')
        ret, frame1 = cap.read()
        if ret is False:
            print("can't capture video")
            return 
        frame1 = cv2.resize(frame1, self.size)
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        while(1):
            ret, frame2 = cap.read()
            if ret is False:
                break
            count_frame += 1
            frame2 = cv2.resize(frame2, self.size)
            # if not single_frame_picked:
            #     if np.random.randn() > 0.75:
            #         self.spatial_frame = frame2
            #         single_frame_picked = True
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')
            if count_frame == 1:
                self.optical_flow = np.stack((horz, vert))
                self.spatial_frames = np.expand_dims(frame2, axis = 0)
            else:
                self.optical_flow = np.concatenate((self.optical_flow, np.expand_dims(horz, 0), np.expand_dims(vert, 0)))
                self.spatial_frames = np.concatenate((self.spatial_frames, np.expand_dims(frame2, axis = 0)))
            prvs = next
        cap.release()

        print(self.optical_flow.shape, self.spatial_frames.shape)
        # step = count_frame/self.optical_flow_channels
        # self.optical_flow = optical_flow[np.array(range(2*count_frame))/2 %step ==0]
        # self.optical_flow = optical_flow[count_frame/2 - self.optical_flow_channels: count_frame/2 + self.optical_flow_channels]

    