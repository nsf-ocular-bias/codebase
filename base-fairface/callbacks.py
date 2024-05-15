import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(
            self.model.optimizer._decayed_lr(tf.float32))})
            # self.model.optimizer._optimizer._decayed_lr(tf.float32))})
        super().on_epoch_end(epoch, logs)



