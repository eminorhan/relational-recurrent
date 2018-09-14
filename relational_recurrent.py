'''Implements a relational LSTM model. Based on the Keras LSTM class.
'''
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Layer, Dense, Embedding, RNN
from keras.legacy import interfaces
from keras import activations, initializers, regularizers, constraints
from keras import backend as K

class RLSTMCell(Layer):
    """Cell class for the RLSTM layer.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the recurrent state.
    """
    def __init__(self, memdim,
                 memsize,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 mlp_activation='relu',
                 use_bias=True,
                 kernel_initializer='he_normal',
                 recurrent_initializer='he_normal',
                 bias_initializer='zeros',
                 unit_forget_bias=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):

        super(RLSTMCell, self).__init__(**kwargs)
        self.memdim = memdim  # D
        self.memsize = memsize  # K
        self.units = self.memdim * self.memsize  # Total number of "units" as in the paper.
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.mlp_activation = activations.get(mlp_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.memdim * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.memdim, self.memdim * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.W_q = self.add_weight(shape=(self.memdim, self.memdim),
                                   name='W_q',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.W_k = self.add_weight(shape=(self.memdim, self.memdim),
                                   name='W_k',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.W_v = self.add_weight(shape=(self.memdim, self.memdim),
                                   name='W_v',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.mlp_kernel_1 = self.add_weight(shape=(self.memdim, self.memdim),
                                   name='mlp_kernel_1',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.mlp_kernel_2 = self.add_weight(shape=(self.memdim, self.memdim),
                                   name='mlp_kernel_2',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.mlp_gain_1 = self.add_weight(shape=(self.memdim,),
                                    name='mlp_gain_1',
                                    initializer=initializers.Ones(),
                                    regularizer=None,
                                    constraint=None)

        self.mlp_gain_2 = self.add_weight(shape=(self.memdim,),
                                    name='mlp_gain_2',
                                    initializer=initializers.Ones(),
                                    regularizer=None,
                                    constraint=None)

        self.mlp_bias_1 = self.add_weight(shape=(self.memdim,),
                                    name='mlp_bias_1',
                                    initializer=initializers.Zeros(),
                                    regularizer=None,
                                    constraint=None)

        self.mlp_bias_2 = self.add_weight(shape=(self.memdim,),
                                    name='mlp_bias_2',
                                    initializer=initializers.Zeros(),
                                    regularizer=None,
                                    constraint=None)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.memdim,), *args, **kwargs),
                        initializers.Ones()((self.memdim,), *args, **kwargs),
                        self.bias_initializer((self.memdim,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.memdim * 3,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.memdim]
        self.kernel_f = self.kernel[:, self.memdim: self.memdim * 2]
        self.kernel_o = self.kernel[:, self.memdim * 2:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.memdim]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.memdim: self.memdim * 2]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.memdim * 2:]

        if self.use_bias:
            self.bias_i = self.bias[:self.memdim]
            self.bias_f = self.bias[self.memdim: self.memdim * 2]
            self.bias_o = self.bias[self.memdim * 2:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_o = None

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if 0 < self.recurrent_dropout < 1 and self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask

        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = K.reshape(states[0], (-1, self.memsize, self.memdim))  # previous hid state: K x D flattened (corresponds to the usual h variables in LSTM)
        c_tm1 = K.reshape(states[1], (-1, self.memsize, self.memdim))  # previous mem state: K x D flattened (corresponds to M in the paper)

        mem_inp_concat = K.concatenate((c_tm1, K.expand_dims(inputs,axis=-2)), axis=-2)

        Q_mat = K.dot(c_tm1, self.W_q)
        K_mat = K.dot(mem_inp_concat, self.W_k)
        V_mat = K.dot(mem_inp_concat, self.W_v)
        c_upd = K.softmax( tf.einsum('ijk,ilk->ijl', Q_mat, K_mat) / self.memdim ** -0.5 )  # softmax axis: -1
        c_upd = tf.einsum('ijk,ikl->ijl', c_upd, V_mat)  # updated memory: K x D (corresponds to M tilde in the paper)

        # MLP-LayerNorm combo
        ici_1 = K.dot(c_upd, self.mlp_kernel_1)
        c_upd = self.mlp_gain_1 * (ici_1 - K.mean(ici_1, axis=-1, keepdims=True)) / (K.std(ici_1, axis=-1, keepdims=True) + 1e-9) + self.mlp_bias_1
        ici_2 = K.dot(self.mlp_activation(c_upd), self.mlp_kernel_2)
        c_upd = self.mlp_gain_2 * (ici_2 - K.mean(ici_2, axis=-1, keepdims=True)) / (K.std(ici_2, axis=-1, keepdims=True) + 1e-9) + self.mlp_bias_2


        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_o = inputs * dp_mask[2]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_o = inputs

        x_i = K.dot(inputs_i, self.kernel_i)
        x_f = K.dot(inputs_f, self.kernel_f)
        x_o = K.dot(inputs_o, self.kernel_o)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_f = K.bias_add(x_f, self.bias_f)
            x_o = K.bias_add(x_o, self.bias_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_o = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_o = h_tm1

        i = self.recurrent_activation(x_i + K.permute_dimensions(K.dot(h_tm1_i, self.recurrent_kernel_i), (1,0,2)))
        f = self.recurrent_activation(x_f + K.permute_dimensions(K.dot(h_tm1_f, self.recurrent_kernel_f), (1,0,2)))
        o = self.recurrent_activation(x_o + K.permute_dimensions(K.dot(h_tm1_o, self.recurrent_kernel_o), (1,0,2)))
        c = f * K.permute_dimensions(c_tm1, (1,0,2)) + i * K.permute_dimensions( c_upd, (1,0,2))

        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        # Swap dimensions back
        h = K.permute_dimensions(h, (1,0,2))
        c = K.permute_dimensions(c, (1,0,2))

        # Reshape states
        h = K.reshape(h, (-1, self.units))
        c = K.reshape(c, (-1, self.units))

        return h, [c, h]

    def get_config(self):
        config = {'memdim': self.memdim,
                  'memsize': self.memsize,
                  'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(RLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RLSTM(RNN):
    """Relational Long Short-Term Memory layer (Santoro et al.)
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """
    #@interfaces.legacy_recurrent_support
    def __init__(self, memdim,
                 memsize,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):

        cell = RLSTMCell(memdim=memdim,
                         memsize=memsize,
                         activation=activation,
                         recurrent_activation=recurrent_activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         recurrent_initializer=recurrent_initializer,
                         unit_forget_bias=unit_forget_bias,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         recurrent_constraint=recurrent_constraint,
                         bias_constraint=bias_constraint,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout)

        super(RLSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(RLSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def memdim(self):
        return self.cell.memdim

    @property
    def memsize(self):
        return self.cell.memsize

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'memdim': self.memdim,
                  'memsize': self.memsize,
                  'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}

        base_config = super(RLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))