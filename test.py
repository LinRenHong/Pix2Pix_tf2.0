

import os
import argparse
import datetime
import tensorflow as tf

from utils.utils import load_image_test
from utils.trainer import ModelCompiler


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model_path", type=str, default=r"YOUR_MODEL_PATH", help="model path")
    args = parser.parse_args()

    PATH = os.path.join("dataset", 'facades/')

    BUFFER_SIZE = 400
    BATCH_SIZE = 1

    # Test dataset
    test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    EPOCHS = 200

    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Checkpoint name
    save_ckpt_name = f"{today}_Pix2Pix"

    # Training configuration & hyper-parameters
    training_configuration = {

        "load_model_path": args.load_model_path,
        "batch_size": BATCH_SIZE,
        "test_dataset": test_dataset,

    }

    trainer = ModelCompiler(**training_configuration)
    trainer.test()
