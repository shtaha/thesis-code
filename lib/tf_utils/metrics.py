import tensorflow as tf

from lib.data_utils import is_nonetype


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    """
        Implementation following: https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef.
    """

    def __init__(self, threshold=0.5, name="mcc", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tps = tf.keras.metrics.TruePositives(thresholds=threshold, name="mcc-tp")
        self.fps = tf.keras.metrics.FalsePositives(thresholds=threshold, name="mcc-fp")
        self.tns = tf.keras.metrics.TrueNegatives(thresholds=threshold, name="mcc-tn")
        self.fns = tf.keras.metrics.FalseNegatives(thresholds=threshold, name="mcc-fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tps.update_state(y_true, y_pred, sample_weight)
        self.fps.update_state(y_true, y_pred, sample_weight)
        self.tns.update_state(y_true, y_pred, sample_weight)
        self.fns.update_state(y_true, y_pred, sample_weight)

    def result(self):
        tp = self.tps.result()
        fp = self.fps.result()
        tn = self.tns.result()
        fn = self.fns.result()

        numerator = tp * tn - fp * fn
        denominator = tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        result = numerator / denominator
        return result

    def reset_states(self):
        self.tps.reset_states()
        self.fps.reset_states()
        self.tns.reset_states()
        self.fns.reset_states()


class BinaryCrossentropy(tf.keras.metrics.Metric):
    def __init__(self, from_logits=False, name="bce", **kwargs):
        super(BinaryCrossentropy, self).__init__(name=name, **kwargs)

        self.from_logits = from_logits

        self.bce_sum = tf.constant(0.0, dtype=tf.float64)
        self.n = tf.constant(0.0, dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        bces = tf.keras.backend.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )

        if not is_nonetype(sample_weight):
            bces = tf.multiply(bces, sample_weight)

        bce_sum = tf.reshape(tf.reduce_sum(bces, axis=-1), [])
        n = tf.cast(tf.shape(y_true)[0], dtype=tf.float64)

        self.bce_sum = tf.math.add(self.bce_sum, bce_sum)
        self.n = tf.math.add(self.n, n)

    def result(self):
        result = tf.math.divide(self.bce_sum, self.n + 1e-9)
        return result

    def reset_states(self):
        self.bce_sum = tf.constant(0.0, dtype=tf.float64)
        self.n = tf.constant(0.0, dtype=tf.float64)
