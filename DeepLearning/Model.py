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

class ValueNet(BaseModel):
    def __init__(self, w, h, c):
        super(ValueNet, self).__init__(w, h, c)
        self.num_res_block = 3  # 設定 ResNet 層數

        # 定義一些常見層
        self.reshape = layers.Reshape((self.w, self.h, self.c))  # 改變輸入形狀

        self.conv_reshape = layers.Conv2D(filters=128,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          use_bias=False,
                                          kernel_regularizer=keras.regularizers.l2(1e-4))

        # 殘差網絡中的層
        self.conv = [layers.Conv2D(filters=128,
                                   kernel_size=5,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(1e-4))
                     for _ in range(2 * self.num_res_block)]  # 使用 2 倍的 ResNet 層數

        self.batch_norm = [layers.BatchNormalization(axis=3)
                           for _ in range(2 * self.num_res_block)]  # 批量正規化

        self.actfun = [layers.Activation('relu')
                       for _ in range(2 * self.num_res_block)]  # 激活函數

        # 全局平均池化層
        self.gap = layers.GlobalAveragePooling2D()

        # 最後的輸出層，用 tanh 激活函數
        self.v = layers.Dense(1, activation='tanh', name='v')  # 預測 [-1, 1] 範圍的值

    def resnet_layer(self, inputs, activation=False, batch_normalization=True, conv_first=True, cnt=0):
        x = inputs
        if conv_first:
            x = self.conv[cnt](x)  # 卷積層
            if batch_normalization:
                x = self.batch_norm[cnt](x)  # 批量正規化
            if activation:
                x = self.actfun[cnt](x)  # 激活函數

        else:
            if batch_normalization:
                x = self.batch_norm[cnt](x)
            if activation:
                x = self.actfun[cnt](x)
            x = self.conv[cnt](x)

        return x

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, activation=True, cnt=i*2)
            resnet = self.resnet_layer(inputs=resnet, cnt=i*2+1)

            # 如果形狀不同，則調整形狀
            if x.shape != resnet.shape:
                x = self.conv_reshape(x)

            resnet = layers.add([resnet, x])  # 殘差連接
            resnet = self.actfun[i*2+1](resnet)  # 激活
            x = resnet

        return x

    def call(self, x):
        x = self.reshape(x)  # 重塑輸入
        x = self.resnet_v1(inputs=x, num_res_blocks=self.num_res_block)  # 殘差網絡
        x = self.gap(x)  # 全局平均池化
        value = self.v(x)  # 輸出預測的值
        return value

    def build_graph(self):
        x = layers.Input(shape=(self.w, self.h, self.c))
        return keras.Model(inputs=[x], outputs=self.call(x))
