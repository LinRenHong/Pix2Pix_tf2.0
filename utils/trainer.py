

import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from models.pix2pix import Generator, Discriminator
from utils.loss import generator_loss, discriminator_loss
from utils.utils import print_progress_bar


class ModelCompiler(object):

    def __init__(self, **kwargs):

        self.kwargs = kwargs

        self.batch_size = kwargs.get("batch_size", None)
        self.epochs = kwargs.get("epochs", None)

        self.save_ckpt_name = kwargs.get("save_ckpt_in_path", None)
        self.tb_log_path = kwargs.get("tensorboard_path", None)

        # Training dataset
        self.train_dataset = kwargs.get("train_dataset", None)
        if self.train_dataset is not None:
            self.train_dataset_size = tf.data.experimental.cardinality(self.train_dataset).numpy()

        # Test dataset
        self.test_dataset = kwargs.get("test_dataset", None)
        if self.test_dataset is not None:
            self.test_dataset_size = tf.data.experimental.cardinality(self.test_dataset).numpy()

        self.results_dir = "results"
        self.save_images_dir = os.path.join(self.results_dir, "images")
        self.save_models_dir = os.path.join(self.results_dir, "saved_models")

        print("\n############################ Generator ############################\n")
        self.generator = Generator()
        self.generator.build(input_shape=(self.batch_size, 256, 256, 3))
        self.generator.summary()

        print("\n############################ Discriminator ############################\n")
        self.discriminator = Discriminator()
        self.discriminator.build(input_shape=[(self.batch_size, 256, 256, 3), (self.batch_size, 256, 256, 3)])
        self.discriminator.summary()

        self.generator_optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(2e-4, beta_1=0.5))
        self.discriminator_optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(2e-4, beta_1=0.5))

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.save_images_dir, exist_ok=True)
        os.makedirs(self.save_models_dir, exist_ok=True)

    def train(self):

        os.makedirs(os.path.join(self.save_models_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_images_dir, "%s" % self.save_ckpt_name), exist_ok=True)

        self.writer = tf.summary.create_file_writer(os.path.join(self.results_dir, self.tb_log_path) + "/train")

        for epoch_idx in range(1, self.epochs + 1):

            # for example_input, example_target in self.test_dataset.take(1):
            #     generate_images(generator, example_input, example_target)

            # Use metrics to storage log history
            gen_total_loss_metrics = tf.keras.metrics.Mean("gen_total_loss")
            gen_gan_loss_metrics = tf.keras.metrics.Mean("gen_gan_loss")
            gen_l1_loss_metrics = tf.keras.metrics.Mean("gen_l1_loss")
            disc_loss_metrics = tf.keras.metrics.Mean("disc_loss")

            # Train
            for batch_idx, (input_image, target) in self.train_dataset.enumerate():
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target)

                # Mean the loss
                gen_total_loss = gen_total_loss.numpy().mean()
                gen_gan_loss = gen_gan_loss.numpy().mean()
                gen_l1_loss = gen_l1_loss.numpy().mean()
                disc_loss = disc_loss.numpy().mean()

                # Use metrics to storage the loss history and mean
                gen_total_loss_metrics(gen_total_loss)
                gen_gan_loss_metrics(gen_gan_loss)
                gen_l1_loss_metrics(gen_l1_loss)
                disc_loss_metrics(disc_loss)

                # Print training step log
                prefix_log = 'Train -> [Epoch: {}/{}] | [Batch: {}/{}]'.format(epoch_idx,
                                                                               self.epochs,
                                                                               batch_idx,
                                                                               self.train_dataset_size + 1)

                suffix_log = "[gen_total_loss: {:1.5f}" \
                             " | gen_gan_loss {:1.5f}" \
                             " | gen_l1_loss {:1.5f}" \
                             " | disc_loss {:1.5f}]".format(float(gen_total_loss),
                                                            float(gen_gan_loss),
                                                            float(gen_l1_loss),
                                                            float(disc_loss))

                print_progress_bar(iteration=int(batch_idx),
                                   total=int(self.train_dataset_size + 1),
                                   prefix=prefix_log,
                                   suffix=suffix_log)

            # Save image
            for example_input, example_target in self.test_dataset.take(1):
                self.generate_and_save_images(example_input, example_target, epoch_done=epoch_idx)

            ###################
            ### TensorBoard ###
            ###################

            # Use metrics result write to tensorboard
            write_to_tensorboard_data = {"gen_total_loss": gen_total_loss_metrics.result(),
                                         "gen_gan_loss": gen_gan_loss_metrics.result(),
                                         "gen_l1_loss": gen_l1_loss_metrics.result(),
                                         "disc_loss": disc_loss_metrics.result()}

            self.write_to_tensorboard(data_dict=write_to_tensorboard_data, epoch_done=epoch_idx, condition='train')

            # Save model
            self.save_model(epoch_done=epoch_idx)

    @tf.function
    def train_step(self, input_image, target):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def save_model(self, epoch_done):

        # Save models checkpoints
        if epoch_done % 5 == 0:
            print("\nSave models to [%s] at %d epoch\n" % (self.save_ckpt_name, epoch_done))

            final_save_model_path = "%s/%s/pix2pix_generator_%s" % (self.save_models_dir, self.save_ckpt_name, epoch_done)
            self.generator.save_weights(final_save_model_path + ".h5")

        # Save latest models
        if epoch_done == self.epochs:
            print("\nSave latest models to [%s]\n" % self.save_ckpt_name)

            final_save_model_path = "%s/%s/pix2pix_generator_%s" % (self.save_models_dir, self.save_ckpt_name, self.epochs)
            self.generator.save_weights(final_save_model_path + ".h5")

    def write_to_tensorboard(self, data_dict, epoch_done, condition):

        if self.writer is not None:

            if condition.strip() in ["train", "Train", "TRAIN"]:
                with self.writer.as_default():
                    for (name, data) in data_dict.items():
                        tf.summary.scalar(name=name, data=data, step=epoch_done)

            else:
                print("Please specify condition: [\"train\", \"Train\", \"TRAIN\"]")
        else:
            print("Writer is None!")

    def generate_and_save_images(self, test_input, tar, epoch_done):

        prediction = self.generator(test_input, training=False)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        final_save_image_path = "%s/%s/epoch_%s" % (self.save_images_dir, self.save_ckpt_name, epoch_done)
        plt.savefig(final_save_image_path + ".png")
        plt.close('all')
