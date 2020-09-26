import tensorflow as tf


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
