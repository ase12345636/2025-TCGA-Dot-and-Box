from keras import layers
import keras
import einops


class BaseModel(keras.Model):
    def __init__(self, w, h, c):
        super(BaseModel, self).__init__()

        self.w = w
        self.h = h
        self.c = c

class ResNet(BaseModel):
    def __init__(self, w, h, c):
        super(ResNet, self).__init__(w, h, c)

        # self.num_res_block = 3
        self.num_res_block = 9
        
        self.reshape = layers.Reshape((self.w, self.h) + (1,))

        self.conv = [layers.Conv2D(filters=128,
                                   kernel_size=5,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(1e-4))
                     for i in range(2*self.num_res_block)]

        self.batch_norm = [layers.BatchNormalization(axis=3)
                           for i in range(2*self.num_res_block)]

        self.actfun = [layers.Activation('relu')
                       for i in range(2*self.num_res_block)]

        self.gap = layers.GlobalAveragePooling2D()
        self.pi = layers.Dense(w*h, activation='softmax', name='pi')

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, activation=True, cnt=i*2)
            resnet = self.resnet_layer(inputs=resnet, cnt=i*2+1)
            resnet = layers.add([resnet, x])
            resnet = self.actfun[i*2+1](resnet)
            x = resnet

        return x

    def resnet_v2(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, activation=True, cnt=i*2)
            resnet = self.resnet_layer(inputs=resnet, cnt=i*2+1)
            resnet = layers.add([resnet, x])
            resnet = self.actfun[i*2+1](resnet)
            x = resnet

        return x
    
    def resnet_layer(self, inputs, activation=False, batch_normalization=True, conv_first=True, cnt=0):

        x = inputs
        if conv_first:
            x = self.conv[cnt](x)
            if batch_normalization:
                x = self.batch_norm[cnt](x)
            if activation:
                x = self.actfun[cnt](x)

        else:
            if batch_normalization:
                x = self.batch_norm[cnt](x)
            if activation:
                x = self.actfun[cnt](x)
            x = self.conv[cnt](x)

        return x

    def call(self, x):
        x = self.reshape(x)
        x = self.resnet_v1(inputs=x,
                           num_res_blocks=self.num_res_block)
        x = self.gap(x)
        predict = self.pi(x)

        return predict

    def build_graph(self):
        x = layers.Input(shape=(self.w, self.h))
        return keras.Model(inputs=[x], outputs=self.call(x))

class C2P1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)


class ResidualMain(keras.layers.Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            C2P1D(filters=filters,
                  kernel_size=kernel_size,
                  padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            C2P1D(filters=filters,
                  kernel_size=kernel_size,
                  padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    """
      Project certain dimensions of the tensor as the data is passed through different 
      sized filters and downsampled. 
    """

    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Conv3D(filters=units,
                          kernel_size=(1, 1, 1),
                          padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    """
      Add residual blocks to the model. If the last dimensions of the input data
      and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters,
                       kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
          Use the einops library to resize the tensor.  

          Args:
            video: Tensor representation of the video, in the form of a set of frames.

          Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos


def get_model(inputs_shape):
    # Input Shape : frame, height, width, channel
    input_shape = inputs_shape
    input = layers.Input(shape=(input_shape))
    x = input

    x = C2P1D(filters=16, kernel_size=(2, 5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(32 // 2, 32 // 2)(x)
    x = layers.Dropout(0.3)(x)

    x = add_residual_block(x, 32, (2, 3, 3))
    x = ResizeVideo(32 // 4, 32 // 4)(x)
    x = layers.Dropout(0.3)(x)

    x = add_residual_block(x, 32, (2, 3, 3))
    x = ResizeVideo(32 // 8, 32 // 8)(x)
    x = layers.Dropout(0.3)(x)

    x = add_residual_block(x, 64, (3, 3, 3))
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(input_shape[1]*input_shape[2],
                     activation='softmax', name='pi')(x)

    model = keras.Model(input, x)

    return model