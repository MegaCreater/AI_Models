# function to define inception layer
def inception_layer(input_layer,k):
    # Single conv layer of kernel size 1x1
    out_1X1_conv = Conv2D(filters=k,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu',kernel_regularizer=None,bias_regularizer=None)(input_layer)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_1X1_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_1X1_conv)
    out_1X1_conv = Activation('relu')(out_1X1_conv)

    # First input is operated with conv layer of kernel size 1x1 
    out_1x1_con_N_3X3_conv = Conv2D(filters=k,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(input_layer)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_1x1_con_N_3X3_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_1x1_con_N_3X3_conv)
    out_1x1_con_N_3X3_conv = Activation('relu')(out_1x1_con_N_3X3_conv)
    # After above (out_1x1_con_N_3X3_conv, i.e.; conv 1x1) operation on input, it is operated again with conv layer of kernel size 3x3
    out_1x1_con_N_3X3_conv = Conv2D(filters=k,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(out_1x1_con_N_3X3_conv)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_1x1_con_N_3X3_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_1x1_con_N_3X3_conv)
    out_1x1_con_N_3X3_conv = Activation('relu')(out_1x1_con_N_3X3_conv)

    # The input is operated with conv layer of kernel size 1x1 
    out_1x1_con_N_5X5_conv = Conv2D(filters=k,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(input_layer)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_1x1_con_N_5X5_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_1x1_con_N_5X5_conv)
    out_1x1_con_N_5X5_conv = Activation('relu')(out_1x1_con_N_5X5_conv)    
    # After above (out_1x1_con_N_5X5_conv, i.e.; conv 1x1) operation on input, it is operated again with conv layer of kernel size 5x5
    out_1x1_con_N_5X5_conv = Conv2D(filters=k,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(out_1x1_con_N_5X5_conv)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_1x1_con_N_5X5_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_1x1_con_N_5X5_conv)
    out_1x1_con_N_5X5_conv = Activation('relu')(out_1x1_con_N_5X5_conv)
    
    # The input is operated with MaxPooling layer of kernel size 3x3
    out_3x3_max_N_1x1_conv = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_layer)
    # After above (out_3x3_max_N_1x1_conv, i.e.; max 3x3) operation on input, it is operated again with conv layer of kernel size 1x1
    out_3x3_max_N_1x1_conv = Conv2D(filters=k,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu',kernel_regularizer=None,bias_regularizer=None)(out_3x3_max_N_1x1_conv)
    # Applying Batch nomalization to reduce number of epochs
    # This layer is optional it can be neglected, as it increases overall traning time
    out_3x3_max_N_1x1_conv = BatchNormalization(axis=-1,epsilon=0.001)(out_3x3_max_N_1x1_conv)
    out_3x3_max_N_1x1_conv = Activation('relu')(out_3x3_max_N_1x1_conv)

    # adding up all the results.
    concatenate_result = concatenate([out_1X1_conv,out_1x1_con_N_3X3_conv,out_1x1_con_N_5X5_conv,out_3x3_max_N_1x1_conv],axis=-1)
    # returning result
    return concatenate_result


# import the necessary modules
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


# main
def googLeNet(width,height,depth,classes):
    '''
        width = width of input image [224 - bestfit].
        height = height of input image [224 - bestfit].
        depth = channels in input image [3 - bestfit](like RGB).
        classes = number of categories in which to classify data [1000 - in original paper].
    '''
    # shape input = (224,224,3) # this is default shape for best reault.
    # One can use any shape of image, but for better result use image of square shape
    # and having dimension like 32, 64, 112, 224 etc; which are many times divisible by 2
    input_layer = Input(shape=(width,height,depth),name='Input_layer')
    model_X_temp = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(input_layer)
    # output shape = (112,112,64)
    model_X_temp = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(model_X_temp)
    # output shape = (56,56,64)
    model_X_temp = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=None,bias_regularizer=None)(model_X_temp)
    # output shape = (56,56,192)
    model_X_temp = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(model_X_temp)
    # output shape = (28,28,192)
    
    # Block 1
    model_X_temp = inception_layer(model_X_temp,64)
    # Leaving  10% neurons in network to prevent overfitting of network.
    # Optional layer as not listed in original paper.
    model_X_temp = Dropout(rate=0.1)(model_X_temp)
    # output shape = (28,28,256)

    # Block 2
    model_X_temp = inception_layer(model_X_temp,120)
    # Optional layer - dropout @ 10%
    model_X_temp = Dropout(rate=0.1)(model_X_temp)
    # output shape = (28,28,480)

    model_X_temp = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(model_X_temp)
    # output shape = (14,14,480)
    
    # Block 3
    model_X_temp = inception_layer(model_X_temp,128)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (14,14,512)
    
    # Block 4
    model_X_temp = inception_layer(model_X_temp,128)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (14,14,512)
    
    # Block 4
    model_X_temp = inception_layer(model_X_temp,128)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (14,14,512)
    
    # Block 4
    model_X_temp = inception_layer(model_X_temp,132)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (14,14,528)

    # Block 4
    model_X_temp = inception_layer(model_X_temp,208)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (14,14,832)

    model_X_temp = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(model_X_temp)
    # output shape = (7,7,832)

    # Block 4
    model_X_temp = inception_layer(model_X_temp,208)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (7,7,832)

    # Block 4
    model_X_temp = inception_layer(model_X_temp,256)
    # Optional layer - dropout @ 20%
    model_X_temp = Dropout(rate=0.2)(model_X_temp)
    # output shape = (7,7,1024)

    model_X_temp = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid')(model_X_temp)
    # output shape = (1,1,1024)
    
    # dropout @ 40%
    model_X_temp = Dropout(rate=0.4)(model_X_temp)
    # Flatten layer - optional; important if using different size images. 
    model_X_temp = Flatten()(model_X_temp)
    # Fully connected layer
    model_X_temp = Dense(units=1000,activation='relu',kernel_regularizer=None,bias_regularizer=None)(model_X_temp)
    # Classification layer
    model_X_temp = Dense(units=classes,activation='softmax',name='Output_layer')(model_X_temp)

    # return built model
    return Model(inputs=input_layer,outputs=model_X_temp,name='googLeNet_By_MegaMachine')
    
