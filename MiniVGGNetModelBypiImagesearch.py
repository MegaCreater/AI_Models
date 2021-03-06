# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


# MiniVGGNet a sequential model, but converted to a model subclass.
# Source: https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
class MiniVGGNetModel(Model):
    def __init__(self, classes, chanDim=-1):
        # call the parent constructor
        super(MiniVGGNetModel, self).__init__()

        # initialize the layers in the first (CONV => RELU) * 2 => POOL
        # layer set
        self.conv1A = Conv2D(32, (3, 3), padding="same")
        self.act1A = Activation("relu")
        self.bn1A = BatchNormalization(axis=chanDim)
        self.conv1B = Conv2D(32, (3, 3), padding="same")
        self.act1B = Activation("relu")
        self.bn1B = BatchNormalization(axis=chanDim)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        # initialize the layers in the second (CONV => RELU) * 2 => POOL
        # layer set
        self.conv2A = Conv2D(32, (3, 3), padding="same")
        self.act2A = Activation("relu")
        self.bn2A = BatchNormalization(axis=chanDim)
        self.conv2B = Conv2D(32, (3, 3), padding="same")
        self.act2B = Activation("relu")
        self.bn2B = BatchNormalization(axis=chanDim)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense3 = Dense(512)
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)

        # initialize the layers in the softmax classifier layer set
        self.dense4 = Dense(classes)
        self.softmax = Activation("softmax")

    def call(self, inputs):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.bn1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.bn1B(x)
        x = self.pool1(x)

        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(inputs)
        x = self.act2A(x)
        x = self.bn2A(x)
        x = self.conv2B(x)
        x = self.act2B(x)
        x = self.bn2B(x)
        x = self.pool2(x)

        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.do3(x)

        # build the softmax classifier
        x = self.dense4(x)
        x = self.softmax(x)

        # return the constructed model
        return x
