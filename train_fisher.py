from __future__ import print_function
import tensorflow as tf
from dataset import Dataset, Queue_Dataset
import Multi_stream_fisher 
import model
import os
import sys

def train_model(model, n_epoch):
    saver = tf.train.Saver()
    file_ckpt = 'checkpoints_01_02_2018'
    if not os.path.exists(file_ckpt):
        os.mkdir(file_ckpt)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement =True)) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        model.queue.start_queue_threads(sess,coord)

        ckpt = tf.train.latest_checkpoint(file_ckpt)
        if ckpt:
            print(ckpt)
            saver.restore(sess, ckpt)
            print("model restored.")

        model.writer.add_graph(sess.graph)
        step = sess.run(model.global_step)

        for epoch in range(n_epoch):
            if step% 500==499:
                model.learning_rate = model.learning_rate /10
            
            # Train phases
            # =====================================================================================
            total_train_loss = 0.0
            total_train_softmax_loss = 0.0
            total_train_spatials_loss = 0.0
            total_train_spatial_loss =0.0
            total_train_temporals_loss=0.0
            total_train_temporal_loss=0.0
            train_accuracy = 0.0
            batch = 1

            for i in range(model.queue.num_batch_train):
                
                feed_dict = {model.lr_placeholder:model.learning_rate, model.queue.is_training: 1}
                
                l, s_l, sp_l, sps_l, t_l, ts_l, acc, _,_,_,_,_  = sess.run([model.loss, model.softmax_loss,
									    model.spatial_loss, model.spatials_loss,
									    model.temporal_loss, model.temporals_loss,
									    model.accuracy, model.optimizer, 
									    model.spatial_centers_update_op, 
									    model.temporal_centers_update_op,
									    model.spatial_center_update_op,
									    model.temporal_center_update_op], feed_dict = feed_dict)
                
                # Print results after each batch
                print("{:3d}/{:3d}: acc {:0.5f}, softmax_loss {:3.5f}, intra_spatial {:3.5f}, extra_spatial {3.5f}, \\
				intra_temporal {:3.5f}, extra_temporal {:3.5f}, total_loss {:5.5f}".format(batch, model.queue.num_batch_train, acc, l, s_l,
													   sps_l, sp_l, ts_l, t_l, total_train_loss),end='\r')
				step +=1
		#if step %3==2: saver.save(sess, file_ckpt+ '/check_points', global_step = model.global_step)
                batch += 1
                total_train_loss += l
		total_train_softmax_loss +=s_l
		total_train_spatials_loss +=sps_l
		total_train_spatial_loss += sp_l
		total_train_temporals_loss+=ts_l
		total_train_temporal_loss+=t_l
                train_accuracy += acc
            
            train_accuracy = train_accuracy/model.queue.num_batch_train
			
            # Test phase
            # =======================================================================================
		total_test_loss = 0.0
			total_test_softmax_loss =0.0
			total_test_spatials_loss = 0.0
			total_test_spatial_loss = 0.0
			total_test_temporals_loss= 0.0
			total_test_temporal_loss= 0.0
            test_accuracy = 0.0
            batch = 1

            for i in range(model.queue.num_batch_test):
                
                feed_dict = {model.queue.is_training: 0}
                
                l, s_l, sp_l, sps_l, t_l, ts_l, acc  = sess.run([model.loss, model.softmax_loss,
									    model.spatial_loss, model.spatials_loss,
									    model.temporal_loss, model.temporals_loss,
									    model.accuracy], feed_dict = feed_dict)
                
                # Print results after each batch
                print("{:3d}/{:3d}: acc {:0.5f}, softmax_loss {:3.5f}, intra_spatial {:3.5f}, extra_spatial {3.5f}, \\
				intra_temporal {:3.5f}, extra_temporal {:3.5f}, total_loss {:5.5f}".format(batch, model.queue.num_batch_test, acc, l, s_l,
													   sps_l, sp_l, ts_l, t_l, total_train_loss),end='\r')
				step +=1
		#if step %3==2: saver.save(sess, file_ckpt+ '/check_points', global_step = model.global_step)
                batch += 1
                total_test_loss += l
				total_test_softmax_loss +=s_l
				total_test_spatials_loss +=sps_l
				total_test_spatial_loss += sp_l
				total_test_temporals_loss+=ts_l
				total_test_temporal_loss+=t_l
                test_accuracy += acc
            
            test_accuracy = test_accuracy/model.queue.num_batch_test
                
            # Summerize phase
            # =====================================================================================   
            print("Epoch {}: train_loss {}, train_acc {}, test_loss {}, test_acc {}".format(epoch, total_train_loss, train_accuracy, total_test_loss, test_accuracy))

            feed_dict = {model.train_loss_placeholder: total_train_loss, 
						 model.test_loss_placeholder: total_test_loss, 
						 model.train_softmax_loss_placeholder: total_train_softmax_loss,
						 model.train_spatials_loss_placeholder: total_train_spatials_loss,
						 model.train_spatial_loss_placeholder:  total_train_spatial_loss,
						 model.train_temporals_loss_placeholder: total_train_temporals_loss,
						 model.train_temporal_loss_placeholder: total_train_temporal_loss,
						 model.test_softmax_loss_placeholder: total_test_softmax_loss,
						 model.test_spatials_loss_placeholder: total_test_spatials_loss,
						 model.test_spatial_loss_placeholder:  total_test_spatial_loss,
						 model.test_temporals_loss_placeholder: total_test_temporals_loss,
						 model.test_temporal_loss_placeholder: total_test_temporal_loss,
						 model.train_acc_placeholder: train_accuracy,
						 model.test_acc_placeholder: test_accuracy}
            
            summary = sess.run(model.merged_summary, feed_dict = feed_dict)
            model.writer.add_summary(summary, epoch)
            
            saver.save(sess, file_ckpt+ '/check_points', global_step = model.global_step)

def main():
    n_epoch = int(sys.argv[1])
    dataset = Queue_Dataset(n_classes=50, folder=sys.argv[2])
    model = Multi_stream_fisher.Multi_stream_model(queue_data=dataset, h5_path=sys.argv[3], learning_rate=0.1,momentum=0.8) 
    train_model(model, n_epoch)

if __name__ == '__main__':
    main()
