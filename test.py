

import os
import argparse
import tensorflow as tf

from utils.utils import load_image_test
from utils.trainer import ModelCompiler


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model_path", type=str, default=r"YOUR_MODEL_PATH", help="model path")
    args = parser.parse_args()

    dataset_path = os.path.join("dataset", 'facades/')

    BUFFER_SIZE = 400
    BATCH_SIZE = 1

    # Test dataset
    test_dataset = tf.data.Dataset.list_files(dataset_path + 'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Test configuration & hyper-parameters
    test_configuration = {

        "load_model_path": args.load_model_path,
        "batch_size": BATCH_SIZE,
        "test_dataset": test_dataset,

    }

    trainer = ModelCompiler(**test_configuration)
    trainer.test()
