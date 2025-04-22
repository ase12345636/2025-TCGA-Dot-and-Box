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

        self.reshape = layers.Reshape((self.w, self.h, self.c))

        self.conv_reshape = layers.Conv2D(filters=128,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          use_bias=False,
                                          kernel_regularizer=keras.regularizers.l2(1e-4))

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

            if x.shape != resnet.shape:
                x = self.conv_reshape(x)

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
        x = layers.Input(shape=(self.w, self.h, self.c))
        return keras.Model(inputs=[x], outputs=self.call(x))
