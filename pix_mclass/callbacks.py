import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, CosineDecay


class CosineDecayResume(LearningRateSchedule):
    def __init__(
            self,
            initial_learning_rate:float,
            decay_steps:int,
            alpha:float=0.0,
            start_step:int=0,
            warmup_learning_rate_factor:float=None,
            warmup_steps:int = 0,
            name="CosineDecayResume"):
        super().__init__()
        self.start_step = int(start_step)
        self.initial_learning_rate = float(initial_learning_rate)
        self.warmup_learning_rate_factor = warmup_learning_rate_factor
        self.warmup_steps = int(warmup_steps)
        self.cosine_decay = CosineDecay(initial_learning_rate, decay_steps-self.warmup_steps, alpha, name=name)

    def __call__(self, step):
        if self.warmup_steps>0 and self.warmup_learning_rate_factor is not None:
            return tf.cond(step<self.warmup_steps, lambda: self.warmup(step), lambda: self.cosine_decay(step + self.start_step - self.warmup_steps))
        else:
            return self.cosine_decay(step + self.start_step - self.warmup_steps)

    def warmup(self, step):
        target_learning_rate = self.cosine_decay(tf.maximum(tf.cast(0, tf.int32), tf.cast(step + self.start_step - self.warmup_steps, tf.int32)))
        dtype = target_learning_rate.dtype
        warmup_lr = tf.cast(self.warmup_learning_rate_factor, dtype=dtype) * target_learning_rate
        step = tf.cast(step, dtype=dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype=dtype)
        return warmup_lr + (target_learning_rate - warmup_lr) * step / warmup_steps
