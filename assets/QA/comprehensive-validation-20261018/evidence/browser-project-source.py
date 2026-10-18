from __future__ import annotations

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers


@keras.saving.register_keras_serializable(package='NeuroWeave')
class PredictionAggregateMetric(keras.metrics.Metric):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        if mode not in ('mean', 'sum'): raise ValueError(f'Unsupported prediction aggregate mode: {mode}')
        self.mode = mode
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        del y_true
        values = keras.ops.cast(y_pred, self.dtype)
        if sample_weight is not None: values = values * keras.ops.cast(sample_weight, values.dtype)
        self.total.assign_add(keras.ops.sum(values))
        if self.mode == 'mean': self.count.assign_add(keras.ops.cast(keras.ops.size(values), self.dtype))
        else: self.count.assign_add(keras.ops.cast(1, self.dtype))
    def result(self):
        if self.mode == 'sum': return self.total
        return self.total / keras.ops.maximum(self.count, keras.ops.cast(1, self.dtype))
    def get_config(self):
        return {**super().get_config(), 'mode': self.mode}


def build_model() -> keras.Model:
    keras.utils.set_random_seed(42)
    keras.mixed_precision.set_global_policy('float32')
    input = keras.Input(shape=[8], name='input', dtype='float32')
    dense = layers.Dense(units=2, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, lora_rank=None, lora_alpha=None, quantization_config=None, name='dense')(input)
    output = layers.Activation('linear', name='output')(dense)
    model = keras.Model(inputs=[input], outputs=[output], name='neuro_weave_model')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=None, metrics=None)
    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
