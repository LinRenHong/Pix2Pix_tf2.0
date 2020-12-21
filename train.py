

import os
import datetime
import tensorflow as tf

from utils.utils import load_image_train, load_image_test
from utils.trainer import ModelCompiler


if __name__ == "__main__":
    # _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    #
    # path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
    #                                       origin=_URL,
    #                                       extract=True)

    PATH = os.path.join("dataset", 'facades/')

    BUFFER_SIZE = 400
    BATCH_SIZE = 1

    # Training dataset
    train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

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

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,

        "save_ckpt_in_path": save_ckpt_name,

    }

    trainer = ModelCompiler(**training_configuration)
    trainer.train()

