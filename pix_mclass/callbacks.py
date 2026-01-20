import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, CosineDecay
from pix_mclass.layers import ScheduledDropout, ScheduledGradientWeight


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


class ScheduledDropoutCallback(tf.keras.callbacks.Callback):
    """
    Automatically discovers ScheduledDropout layers and updates their progress
    based on training epochs. Each layer handles its own schedule via max_progress.
    """

    def __init__(self, n_epochs, verbose=0):
        """
        Args:
            n_epochs: Total number of training epochs
            verbose: 0, 1, or 2 for logging level
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.dropout_layers = []

    @staticmethod
    def _find_dropout_layers(layer):
        """
        Recursively find all ScheduledDropout layers in a model.

        Args:
            layer: A Keras layer or model

        Returns:
            List of ScheduledDropout layers
        """
        if isinstance(layer, ScheduledDropout):
            return [layer]

        # Handle nested models/layers
        layers = []
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                layers.extend(ScheduledDropoutCallback._find_dropout_layers(sublayer))
        return layers

    def on_train_begin(self, logs=None):
        """Scan model and register all ScheduledDropout layers"""
        self.dropout_layers = self._find_dropout_layers(self.model)

        if self.verbose > 1:
            print(f"\n{'=' * 70}")
            print(f"CurriculumDropoutScheduler initialized")
            print(f"{'=' * 70}")
            print(f"Found {len(self.dropout_layers)} ScheduledDropout layer(s)")
            print(f"Total epochs: {self.n_epochs}")
            print(f"\nLayer configurations:")

            for i, layer in enumerate(self.dropout_layers):
                completion_epoch = int(layer.max_progress * self.n_epochs)
                print(f"  {layer.name}:")
                print(f"    min_rate={layer.min_rate:.3f}, max_rate={layer.max_rate:.3f}")
                print(f"    max_progress={layer.max_progress:.3f} → completes at epoch {completion_epoch}")

            print(f"{'=' * 70}\n")

    def on_epoch_begin(self, epoch, logs=None):
        """Update global progress at the beginning of each epoch"""
        if not self.dropout_layers:
            return

        # Global training progress [0, 1]
        global_progress = epoch / self.n_epochs if self.n_epochs > 0 else 1.

        # Update all layers with the same global progress
        # Each layer will compute its own rate based on its max_progress
        for layer in self.dropout_layers:
            layer.set_progress(global_progress)
            if self.verbose > 0 and epoch % 50  == 0:
                print(f"epoch: {epoch} (progress: {global_progress}) layer: {layer.name} dropout rate: {layer.get_current_rate()}")


class ScheduledGradientCallback(tf.keras.callbacks.Callback):
    """
    Automatically discovers ScheduledGradientWeight layers and updates their progress
    based on training epochs. Each layer handles its own schedule via max_progress.
    """

    def __init__(self, n_epochs, verbose=0):
        """
        Args:
            n_epochs: Total number of training epochs
            verbose: 0, 1, or 2 for logging level
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.gradient_layers = None

    @staticmethod
    def _find_gradient_layers(layer):
        """
        Recursively find all ScheduledGradientWeight layers in a model.

        Args:
            layer: A Keras layer or model

        Returns:
            List of ScheduledGradientWeight layers
        """
        if isinstance(layer, ScheduledGradientWeight):
            return [layer]

        # Handle nested models/layers
        layers = []
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                layers.extend(ScheduledGradientCallback._find_gradient_layers(sublayer))
        return layers

    def on_train_begin(self, logs=None):
        """Scan model and register all ScheduledGradientWeight layers"""
        self.gradient_layers = self._find_gradient_layers(self.model)

        if self.verbose > 1:
            print(f"\n{'=' * 70}")
            print(f"ScheduledGradientCallback initialized")
            print(f"{'=' * 70}")
            print(f"Found {len(self.gradient_layers)} ScheduledGradientWeight layer(s)")
            print(f"Total epochs: {self.n_epochs}")
            print(f"\nLayer configurations:")
            for i, layer in enumerate(self.gradient_layers):
                completion_epoch = int(layer.max_progress * self.n_epochs)
                print(f"  {layer.name}:")
                print(f"    min_weight={layer.min_weight:.3f}, max_weight={layer.max_weight:.3f}")
                print(f"    max_progress={layer.max_progress:.3f} → completes at epoch {completion_epoch}")

            print(f"{'=' * 70}\n")

    def on_epoch_begin(self, epoch, logs=None):
        """Update global progress at the beginning of each epoch"""
        if not self.gradient_layers:
            return
        global_progress = epoch / self.n_epochs if self.n_epochs > 0 else 1.
        # Update all layers with the same global progress, each layer will compute its own weight based on its max_progress
        for layer in self.gradient_layers:
            layer.set_progress(global_progress)
            if self.verbose > 0 and epoch % 50 == 0: # and epoch % 100 == 0
                print(f"epoch: {epoch} layer: {layer.name} gradient weight: {layer.get_current_weight()}")

