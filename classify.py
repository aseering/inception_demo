#!/usr/bin/env python3

import os
import sys

from classify_helpers import NODE_LOOKUP, MODEL_DIR

import numpy as np
import tensorflow as tf


def run_inference_on_image(image):
    """Runs inference on an image.

    Args:
        image: Image file name.

    Returns:
        Tuple of (Human-readable name (str), score (float between 0 and 1))
    """
    with open(image, 'rb') as f:
        image_data = f.read()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                            {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        predictions_sorted_by_fit = predictions.argsort()
        best_prediction = predictions_sorted_by_fit[-1]

        human_readable_name = NODE_LOOKUP.id_to_string(best_prediction)
        score = predictions[best_prediction]

        return (human_readable_name, score)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image of a panda
        image_path = os.path.join(MODEL_DIR, 'cropped_panda.jpg')

    print(run_inference_on_image(image_path))
