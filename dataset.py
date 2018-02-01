import tensorflow as tf
import numpy as np 
import pickle
import os
import math
import random
from threading import Thread
from video import video_Obj
from concurrent.futures import ThreadPoolExecutor, wait

class Dataset():
    def __init__(self, n_classes, folder = 'train_data', size=224, split_rate = 0.75, batch_size = 32):
        self.size = size
        self.batch_size = batch_size
        self.folder = folder
        self.n_classes = n_classes
        self.split_rate = split_rate
        self._load_data()
        
    def _load_data(self):
        self.pkl_list = [os.path.join(self.folder, pkl_file) for pkl_file in os.listdir(self.folder)]
        np.random.shuffle(self.pkl_list)
        
        self.len_data = len(self.pkl_list)
        self.len_data_train = int(self.split_rate * self.len_data)
        self.len_data_test = self.len_data - self.len_data_train

        self.pkl_list_train = self.pkl_list[:self.len_data_train]
        self.pkl_list_test = self.pkl_list[self.len_data_train:]
        self.num_batch_train = int(math.ceil(float(self.len_data_train) / self.batch_size))
        self.num_batch_test = int(math.ceil(float(self.len_data_test) / self.batch_size))


    def process_vid(self, video_path, optical_size):
        def prewhiten(x):
            mean = np.mean(x)
            std = np.std(x)
            std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
            y = np.multiply(np.subtract(x, mean), 1/std_adj)
            return y   
        # with open(video_path, 'rb') as f:
        #     video = pickle.load(f)       
        video = np.load(video_path)      
        n_frames = video['spatial_frames'].shape[0]
        spatial_idx = np.random.randint(n_frames)
        optical_idx = np.random.randint(n_frames - optical_size)
        optical_idxs = np.arange(optical_idx, optical_idx + 2 *optical_size)
        # optical_idxs = np.random.randint(n_frames, size = 10)
        
        # horz = 2*optical_idxs - 1
        # vert = 2*optical_idxs
        # optical_idxs = np.concatenate((horz, vert))
        # optical_idxs.sort()
        
        spatial_frame = video['spatial_frames'][spatial_idx]
        optical_flow = video['optical_flow'][optical_idxs]
        optical_flow = np.swapaxes(optical_flow, 0, 2)
        
        size = spatial_frame.shape[0]
        corner_x = np.random.randint(size-self.size)
        corner_y = np.random.randint(size-self.size)

        crop_spatial_frame = spatial_frame[corner_x : corner_x+self.size,corner_y:corner_y+self.size]
        crop_optical_flow = optical_flow[corner_x : corner_x+self.size,corner_y:corner_y+self.size]
        return prewhiten(crop_spatial_frame), prewhiten(crop_optical_flow), video['labone_hot_label']

    def process_batch(self, videos, optical_size = 10):
        spatial_inputs = []
        temporal_inputs = []
        labels = []
        
        for video in videos:
            spatial_frame, optical_flow, label = self.process_vid(video_path = video, optical_size = optical_size)
            spatial_inputs.append(spatial_frame)
            temporal_inputs.append(optical_flow)
            labels.append(label)
        
        spatial_inputs = np.squeeze(np.array(spatial_inputs))
        temporal_inputs = np.array(temporal_inputs)
        labels = np.array(labels)
        
        return spatial_inputs, temporal_inputs, labels
    
    def next_batch(self, mode = 'train'):
        if mode == 'train':
            pkl_list = self.pkl_list_train
            num_batch = self.num_batch_train
        elif mode == 'test':
            pkl_list = self.pkl_list_test
            num_batch = self.num_batch_test
        np.random.shuffle(pkl_list)
        idx = 0
        start = 0
        pool = ThreadPoolExecutor(1)
        future = pool.submit(self.process_batch, pkl_list[start:start+self.batch_size])
        start += self.batch_size
        while(idx < num_batch - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.process_batch, pkl_list[start:start+self.batch_size])
            yield minibatch
            idx += 1
            start += self.batch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    
# incomplete tf queue
class Queue_Dataset(Dataset):
    def __init__(self, n_classes, folder = 'train_data', size = 224, optical_size = 10, num_threads = 8, batch_size = 32, split_rate = 0.75):
        Dataset.__init__(self, size=size, n_classes = n_classes, 
                        folder = folder,batch_size=batch_size,
                        split_rate = split_rate)
        self.optical_size = optical_size
        self.train_idx = -1
        self.test_idx = -1
        self.num_threads = num_threads
        self._make_inputs()

    def _make_inputs(self):
        self.is_training = tf.placeholder(tf.int32, [])
        self.spatial_input = tf.placeholder(shape = [self.size, self.size, 3], dtype = tf.float32)
        self.temporal_input = tf.placeholder(shape = [self.size, self.size, 2 * self.optical_size], dtype = tf.float32)
        self.one_hot_label = tf.placeholder(shape = [self.n_classes], dtype = tf.float32)

        self.train_queue = tf.RandomShuffleQueue(capacity = self.batch_size*38,
                                        min_after_dequeue = self.batch_size,
                                        dtypes = [tf.float32, tf.float32, tf.float32],
                                        shapes = [[self.size, self.size, 3], [self.size,self.size,2 * self.optical_size], [self.n_classes]],
                                        shared_name = "train_queue")
        
        self.test_queue = tf.RandomShuffleQueue(capacity = self.batch_size*9,
                                        min_after_dequeue = self.batch_size,
                                        dtypes = [tf.float32, tf.float32, tf.float32],
                                        shapes = [[self.size, self.size, 3], [self.size,self.size,2 * self.optical_size], [self.n_classes]],
                                        shared_name = "test_queue")
        self.queue = tf.QueueBase.from_list(self.is_training, [self.test_queue, self.train_queue])

        self.train_enqueue_op = self.train_queue.enqueue([self.spatial_input, self.temporal_input, self.one_hot_label])
        self.test_enqueue_op = self.test_queue.enqueue([self.spatial_input, self.temporal_input, self.one_hot_label])
        
        self._train_queue_close = self.train_queue.close(cancel_pending_enqueues=True)
        self._test_queue_close = self.test_queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        batch_x1, batch_x2, batch_y = self.queue.dequeue_up_to(self.batch_size)
        #batch_x1 = tf.Print(batch_x1, [self.train_queue.size(), self.test_queue.size()], message="Size of 2 queue:")
        return batch_x1, batch_x2, batch_y
    
    def close_queue(self, session):
        session.run(self._queue_close)
        
    def _pre_batch_queue(self, sess, coord, is_training):
        while not coord.should_stop():
            if is_training:
                self.train_idx += 1
                self.train_idx = self.train_idx % self.len_data_train
                path = self.pkl_list_train[self.train_idx]
            else:
                self.test_idx += 1
                self.test_idx = self.test_idx % self.len_data_test
                path = self.pkl_list_test[self.test_idx]

            video = np.load(path)      

            n_frames = video['spatial_frames'].shape[0]
            
            spatial_idx = np.random.randint(n_frames)
            try:
                optical_idx = np.random.randint(n_frames - self.optical_size)
            except:
                optical_idx = 0
            optical_idxs = np.arange(optical_idx, optical_idx + 2 *self.optical_size)
            
            spatial_frame = video['spatial_frames'][spatial_idx]
            optical_flow = video['optical_flow'][optical_idxs].astype(float)
            optical_flow = np.swapaxes(optical_flow, 0, 2)
            
            spatial_frame = np.squeeze(spatial_frame)
            optical_flow = np.squeeze(optical_flow)

            size = spatial_frame.shape[0]
            corner_x = np.random.randint(size-self.size)
            corner_y = np.random.randint(size-self.size)

            crop_spatial_frame = spatial_frame[corner_x : corner_x+self.size,corner_y:corner_y+self.size]
            crop_optical_flow = optical_flow[corner_x : corner_x+self.size,corner_y:corner_y+self.size];mean = np.mean(crop_optical_flow);crop_optical_flow-=mean;
		
            if is_training:
                sess.run(self.train_enqueue_op,feed_dict = {self.spatial_input : crop_spatial_frame,
                                                            self.temporal_input: crop_optical_flow,
                                                            self.one_hot_label: video['labone_hot_label']})
            else:
                sess.run(self.test_enqueue_op,feed_dict = {self.spatial_input : crop_spatial_frame,
                                                            self.temporal_input: crop_optical_flow,
                                                            self.one_hot_label: video['labone_hot_label']})

    def start_queue_threads(self, sess, coord):
        train_queue_threads = [Thread(target = self._pre_batch_queue, args = (sess, coord, True)) 
                                                        for i in range(self.num_threads)]
        
        test_queue_threads = [Thread(target = self._pre_batch_queue, args = (sess, coord, False)) 
                                                        for i in range(self.num_threads)]

        queue_threads = train_queue_threads + test_queue_threads
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()

        return queue_threads
   
