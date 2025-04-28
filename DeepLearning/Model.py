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

class ValueNet(ResNet):
    def __init__(self, w, h, c):
        super(ValueNet, self).__init__(w, h, c)
        self.num_res_block = 5

        self.reshape = layers.Reshape((self.w, self.h, self.c))

        # 這裡把 pi 層改成 v 層（實數輸出）
        self.v = layers.Dense(1, activation='tanh', name='v')  # 輸出一個 [-1, 1] 之間的數

        self.conv_reshape = layers.Conv2D(filters=128,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          use_bias=False,
                                          kernel_regularizer=keras.regularizers.l2(1e-4))

    
    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, activation=True, cnt=i*2)
            resnet = self.resnet_layer(inputs=resnet, cnt=i*2+1)

            if x.shape != resnet.shape:
                x = self.conv_reshape(x)

            resnet = layers.add([resnet, x])
            resnet = self.actfun[i*2+1](resnet)
            x = resnet

        return x

    def call(self, x):
        x = self.reshape(x)
        x = self.resnet_v1(inputs=x,
                           num_res_blocks=self.num_res_block)
        x = self.gap(x)
        value = self.v(x)

        return value

    def build_graph(self):
        x = layers.Input(shape=(self.w, self.h, self.c))
        return keras.Model(inputs=[x], outputs=self.call(x))

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