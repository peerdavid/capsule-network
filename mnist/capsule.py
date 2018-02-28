from keras import layers, initializers
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_vector, num_routing, **kwargs):
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get('glorot_uniform')

        super(CapsuleLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """ Here we add "all" trainable weight params. The params adjusted
            by the dynamic-routing algorithm are not added here because those
            are adjusted during the forward pass.

            :param input_shape: (None, num_capsules, dim_capsule)
        """
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Create trainable weight variables for this layer.
        # Note: We need a W_ij with i input capsules and j output capsules.
        #       u is of dim (batch_size, num_capsule, dim_capsule) 
        #       so W needs to be of shape (batch_size, num_output_capsule, num_input_capsule, dim_output_capsule, dim_input_capsule)
        #       where batch_size is tiled inside of call(...)
        self.W = self.add_weight(name='WeightMatrix', 
                                      shape=(1, self.num_capsule, self.input_num_capsule,
                                             self.dim_vector, self.input_dim_vector),
                                      initializer='uniform',
                                      trainable=True)

        super(CapsuleLayer, self).build(input_shape)


    def call(self, u, training = False):
        batch_size = tf.shape(u)[0]
        
        # First of all we add one dimension to the input and duplicate it num_capsule times to get the output of the
        # previous layer as input for every capsule of the following layer.
        # Output shape = (batch_size, num_capsule, input_num_capsule, input_dim)
        u_expand = K.expand_dims(u, 1)
        u_tiled = K.tile(u_expand, [1, self.num_capsule, 1, 1])

        # Now we want to weight the input via u*W
        # So we expand and tile our weight matrix W into the batch_size dimension 
        # such that we are able to multiply W with u_hat
        # Note: This is much faster than k.map_fn
        W_tiled = K.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = K.batch_dot(u_tiled, W_tiled, [3,4])

        # Initialize the log prior probabilities with zero
        b_ij = tf.zeros(shape=[K.shape(u_hat)[0], self.num_capsule, self.input_num_capsule])

        # Start with the dynamic routing algorithm
        for i in range(self.num_routing):
            c_ij = tf.nn.softmax(b_ij, dim=1)
            s_j = K.batch_dot(c_ij, u_hat, [2, 2])
            v_j = squashing(s_j)
            b_ij += K.batch_dot(v_j, u_hat, [2, 3])

        return v_j


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsule, self.dim_vector)



def PrimaryCaps(layer_input, name, dim_capsule, channels, kernel_size=9, strides=2, padding='valid'):
    """ PrimaryCaps layer can be seen as a convolutional layer with a different 
        activation function (squashing)

        :param layer_input
        :param name
        :param dim_capsule
        :param channels
        :param kernel_size 
    """
    assert channels % dim_capsule == 0, "Invalid size of channels and dim_capsule"

    # I.e. each primary capsule contains 8 convoutional units with a 9x9 kernel and a stride of 2.
    num_filters = channels * dim_capsule
    conv_layer = layers.Conv2D(
        name=name, 
        filters=num_filters, 
        kernel_size=kernel_size, 
        strides=strides, 
        activation=None,    # We apply squasing later, therefore no activation funciton is needed here
        padding=padding)(layer_input)

    # In total PrimaryCapsules has [32x6x6] capsule outputs (each outpus is an 8D vector) and each
    # capsule in the [6x6] grid is sharing their weights with each other
    # See https://keras.io/layers/core/#reshape
    reshaped_conv = layers.Reshape(target_shape=(-1, dim_capsule))(conv_layer)

    # Now lets apply the squashing function
    return layers.Lambda(squashing)(reshaped_conv)


def squashing(vectors, axis=-1):
    """ Nonlinear squashing function - Short vectors shrunk to almost 0, long vectors to a length slightly below 1.
        :param vectors: Multiple vectors of one single layer of input shape (None, n, d) with n vectors of dimension d.
        :param axis: Axis of shape (0,1,2) to sqash
        :return Same shape as input (None, n, d) but squashed to 0 or unit vectors
    """

    # Shape (None, n) if keepdims=False, so keep the dim otherwise we cannot squash in the next line ;)
    vector_squared_norm = K.sum(K.square(vectors), axis=axis, keepdims=True)

    # As in [1] but with numerical stability by adding epsilon
    return (vector_squared_norm / (1 + vector_squared_norm)) * (vectors / K.sqrt(vector_squared_norm + K.epsilon()))


def margin_loss(Tk, v_norm):
    """ As defined in [1] eq(4)
    """
    m_plus = 0.9
    m_minus = 0.1
    down_weighting = 0.5

    Lk = (Tk * K.square(K.maximum(0., m_plus - v_norm))) + \
         down_weighting * ((1 - Tk) * K.square(K.maximum(0., v_norm - m_minus)))

    # The total loss is simply the sum of the losses of all digit capsules
    L = K.sum(Lk, axis=1)

    return L


def reconstruction_loss(y_true, y_pred):
    """ We minimize the sum of squared differences between the outputs of the logistic
        units and the pixel intensities.
    """
    return K.sum(K.square(y_pred - y_true ), axis=-1)


#
# HELPERS
#
class Length(layers.Layer):
    """
    See https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    See https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])