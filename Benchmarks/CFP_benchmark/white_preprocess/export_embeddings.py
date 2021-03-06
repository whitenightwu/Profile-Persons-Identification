"""
Exports the embeddings and labels of a directory of images as numpy arrays.

Typicall usage expect the image directory to be of the openface/facenet form and
the images to be aligned. Simply point to your model and your image directory:
    python facenet/contributed/export_embeddings.py ~/models/facenet/20170216-091149/ ~/datasets/lfw/mylfw

Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names, --labels_strings_name to change name


Use --image_batch to dictacte how many images to load in memory at a time.

If your images aren't already pre-aligned, use --is_aligned False

I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done use the facenet library if the image
is pre-aligned. If the image isn't pre-aligned, I use the compare.py function.
I've found working with the embeddings useful for classifications models.

Charles Jekel 2017

"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet


from six.moves import xrange

def main(args):
    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(args.data_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if \
       os.path.isdir(os.path.join(path_exp, name))]

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
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches -1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size

                # Get images for the batch
                images = facenet.load_data(image_list[i*batch_size:n], False, False, args.image_size)
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }

                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            #   export emedings and labels
            label_list  = np.array(label_list)
            np.save(args.embeddings_name, emb_array)
            np.save(args.labels_name, label_list)
            label_strings = np.array(label_strings)
            np.save(args.labels_strings_name, label_strings[label_list])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='/home/ydwu/project3/zihui_DREAM/embedding_facenet/squeezenet',
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('--data_dir', type=str,
                        default='/home/ydwu/project3/zihui_DREAM/embedding_facenet/white-lfw',
        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')


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
