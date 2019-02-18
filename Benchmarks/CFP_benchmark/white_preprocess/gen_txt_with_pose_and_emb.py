from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator



def main(args):
    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(args.data_dir)

    classes_name = []
    img_name = []
    for path in os.listdir(path_exp):
        # if os.path.isdir(path):
        for tpath in os.listdir(os.path.join(path_exp, path)):
            classes_name.append(path)
            img_name.append(tpath)

    # nrof_images = len(image_list)
    # classes_img_name = np.zeros((nrof_images, 2))
    # iii = 0
    # for path in os.listdir(path_exp):
    #     for tpath in os.listdir(os.path.join(path_exp, path)):
    #         classes_img_name[iii,:] = [path, tpath]
    #         iii = iii + 1

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    # sess_2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess_2 = tf.Session()
    my_head_pose_estimator = CnnHeadPoseEstimator(sess_2)  # Head pose estimation object

    # Load the weights from the configuration folders
    my_head_pose_estimator.load_roll_variables("roll/cnn_cccdd_30k.tf")
    my_head_pose_estimator.load_pitch_variables("pitch/cnn_cccdd_30k.tf")
    my_head_pose_estimator.load_yaw_variables("yaw/cnn_cccdd_30k.tf")


    with tf.Graph().as_default():

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


        # with tf.Session() as sess:
        with sess.as_default():

            # Load the model
            facenet.load_model(args.model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = args.image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            pose_array = np.zeros((nrof_images, 3))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches -1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size

                # Get images for the batch
                # images = facenet.load_data(image_list[i*batch_size:n], False, False, args.image_size)
                images, pose = facenet.ydwu_load_data(image_list[i * batch_size:n], False, False, args.image_size, my_head_pose_estimator)
                # print("pose = ", pose)
                pose_array[i * batch_size:n, :] = pose

                feed_dict = { images_placeholder: images, phase_train_placeholder:False }

                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                # print("embed = ", embed)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)


                ydwu_image_list = image_list[i * batch_size:n]

                aaa = pose.tolist()
                bbb = embed.tolist()
                ff = open('/home/ydwu/project3/pose_and_emb.txt', 'a')


                # for jj in range(i * batch_size, n):
                    # ff.write(classes_name[jj] + ' ')
                    # ff.write(img_name[jj] + ' ')

                for jj in range(len(ydwu_image_list)):
                    # ydwu_class = str(ydwu_image_list[jj]).split("/")[-2]
                    # ydwu_img = str(ydwu_image_list[jj]).split("/")[-1]
                    ff.write(str(ydwu_image_list[jj]).split("/")[-2] + ' ')
                    ff.write(str(ydwu_image_list[jj]).split("/")[-1] + ' ')
                    ff.write(str(aaa[jj]).strip('[').strip(']').replace(',', '') + ' ')
                    ff.write(str(bbb[jj]).strip('[').strip(']').replace(',', '') + '\n')

                ff.close()

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            #   export emedings and labels
            label_list  = np.array(label_list)

            # aaa = pose_array.tolist()
            # bbb = emb_array.tolist()

            # #### way1
            # f = open('/home/ydwu/project3/zihui_DREAM/preprocess/pose.txt', 'a')
            # for i in range(nrof_images):
            #     f.write(classes_name[i] + ' ')
            #     f.write(img_name[i] + ' ')
            #     f.write(str(aaa[i]).strip('[').strip(']').replace(',','') + '\n')
            # f.close()

            #### way2
            # f = open('/home/ydwu/project3/pose_and_emb.txt', 'a')
            # for i in range(nrof_images):
            #     f.write(classes_name[i] + ' ')
            #     f.write(img_name[i] + ' ')
            #     f.write(str(aaa[i]).strip('[').strip(']').replace(',','') + ' ')
            #     f.write(str(bbb[i]).strip('[').strip(']').replace(',', '') + '\n')
            #
            # f.close()




            #### way3
            # np.savetxt('/home/ydwu/project3/zihui_DREAM/preprocess/embedding.txt', classes_name, img_name, emb_array)
            # np.savetxt('/home/ydwu/project3/zihui_DREAM/preprocess/pose.txt', pose_array)

            # np.save(args.embeddings_name, emb_array)
            # np.save(args.labels_name, label_list)
            # label_strings = np.array(label_strings)
            # np.save(args.labels_strings_name, label_strings[label_list])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='/home/ydwu/project3/zihui_DREAM/preprocess/squeezenet',
        help='Directory containing the meta_file and ckpt_file')

    parser.add_argument('--data_dir', type=str,
                        default = '/media/ydwu/Document/Datasets/white-ms1mclean',
        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    #default='/home/ydwu/project3/zihui_DREAM/preprocess/white-lfw',
    # default = '/home/ydwu/datasets/white-lfw',
    # default = '/media/ydwu/Document/Datasets/white-ms1mclean',

    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_batch', type=int,
        help='Number of images stored in memory at a time. Default 500.',
        default=77)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)

    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str,
        help='Enter string of which the embeddings numpy array is saved as.',
        default='embeddings.npy')
    parser.add_argument('--labels_name', type=str,
        help='Enter string of which the labels numpy array is saved as.',
        default='labels.npy')
    parser.add_argument('--labels_strings_name', type=str,
        help='Enter string of which the labels as strings numpy array is saved as.',
        default='label_strings.npy')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
