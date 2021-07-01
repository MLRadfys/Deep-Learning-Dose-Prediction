#==============================================================================#
#  Author:       Michael Lempart                                                #
#  Copyright:    2021, Department of Radiation Physics, Lund University                                       #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#

from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import AveragePooling2D
from keras.layers import Input
from keras import backend
from keras import regularizers
from keras import models
from keras.optimizers import Adam


class DenseUNet:

    """

    This class can be used to build a customized Unet which used densely connected blocks as well as transition layers with Average pooling.
    

    """

    def __init__(self, input_shape, dropout_rate = None, l2 = 0.00000001, kernel_initializer = 'he_normal', 
                activation = 'relu', lr = 0.0001, print_summary = False):
        
        """

        Initialzier for the DenseUNet class.

        Args:
            input_shape (tuple) : shape of the input tensor in (width, height, channels).
            dropout_rate (float): dropout value between 0.0 - 1.0.
            l2 (float): L2 regularization value.
            kernel_initializer (string): 'he_normal' as a standard initializer when used with relu activation.
            activation (string): activation function used in the model.
            optimizer (string): Optimizer to be used. Should be either Adam or RMSProp.
            lr (float): learning rate used by the optimizer.
            print_summary (bool): prints the model summary if set to True.

        Returns:
            None    

        """
        self.input_shape = input_shape
        if dropout_rate == 'None':
            self.dropout_rate = None
        self.l2 = l2
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.lr = lr
        self.print_summary = print_summary
    

    def dense_block(self, x, blocks, growth_rate, name, dropout = None):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, growth_rate, name=name + '_block' + str(i + 1), dropout = dropout)
        return x


    def conv_block(self, x, growth_rate, name, dropout = None):
        """
        ,
        Densely connected block like used in DenseNet.

        Args:
            x (tensor): input tensor.
            growth_rate (float): growth rate at dense layers, number of feature maps added.
            name (string): block name.

        Returns:
            x (tensor): Output tensor of the dense block.

        """

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv', kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(x1)
        
        if dropout is not None:        
            x1 = Dropout(self.dropout_rate)(x1)
    

        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)    
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv', kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(x1)
        
        if dropout is not None:        
            x1 = Dropout(self.dropout_rate)(x1)


        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x


    def transition_block(self, x, reduction, name, dropout = None):

        """
        
        Ttransition block using Average Poolning as a downsampling operation.

        Args:
            x (tensor): input tensor.
            reduction (float): compression rate for the transition layers used as in DenseNet.
            name (string): block label.

        Returns:
            x (tensor) : output tensor of the transition block.

        """
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv', kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(1e-7))(x)
        
        if dropout is not None:
            x = Dropout(self.dropout_rate)(x)


        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        
        return x


    def build_DenseUNet(self):

        """
        
        Builds the DenseUNet model.

        Args:
            None   

        Returns:
            Model (obj): Model object    

        """
        
        #Input Layer
        img_input = Input(shape=self.input_shape)

        #Initial convolution
        x1 = Conv2D(32, (3,3), padding='same', use_bias = False, kernel_regularizer= regularizers.l2(self.l2))(img_input)
        x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
        pool_x1 = Activation('relu')(x1)

        #Encoder
        x2 = self.dense_block(pool_x1, 4, 8,'stage_1' ,dropout= self.dropout_rate)
        pool_x2 = self.transition_block(x2, 1.0, 'pool_stage_1')

        x3 = self.dense_block(pool_x2, 4, 16, 'stage_2',dropout= self.dropout_rate)
        pool_x3 = self.transition_block(x3, 1.0, 'pool_stage_2')

        x4 = self.dense_block(pool_x3, 4, 32, 'stage_3',dropout= self.dropout_rate)   
        pool_x4 = self.transition_block(x4, 1.0, 'pool_stage_3')

        x5 = self.dense_block(pool_x4, 4, 64,  'stage_4',dropout= self.dropout_rate)   
        pool_x5 = self.transition_block(x5, 1.0, 'pool_stage_4')

        #Dense Bottleneck
        x6 = self.dense_block(pool_x5, 4, 128,  'stage_5',dropout= None)

        #Decoder
        up1 = Conv2DTranspose(512, (3,3),strides = (2,2), padding = 'same')(x6)
        up1 = concatenate([up1,x5])

        up1 = Conv2D(256, (1,1), padding='same', use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(up1)
        up1 = BatchNormalization(axis=3, epsilon=1.001e-5)(up1)
        up1 = Activation('relu')(up1)
        up1 = self.dense_block(up1, 4, 64, 'stage_6',dropout= self.dropout_rate)

        up2 = Conv2DTranspose(256, (3,3),strides = (2,2), padding = 'same')(up1)
        up2 = concatenate([up2,x4])

        up2 = Conv2D(128, (1,1), padding='same', use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(up2)
        up2 = BatchNormalization(axis=3, epsilon=1.001e-5)(up2)
        up2 = Activation('relu')(up2)
        up2 = self.dense_block(up2, 4, 32,  'stage_7',dropout= self.dropout_rate)

        up3 = Conv2DTranspose(128, (3,3),strides = (2,2), padding = 'same')(up2)
        up3 = concatenate([up3,x3])
        
        up3 = Conv2D(64, (1,1), padding='same', use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(up3)
        up3 = BatchNormalization(axis=3, epsilon=1.001e-5)(up3)
        up3 = Activation('relu')(up3)
        up3 = self.dense_block(up3, 4, 16, 'stage_8',dropout= self.dropout_rate)

        up4 = Conv2DTranspose(64, (3,3),strides = (2,2), padding = 'same')(up3)
        up4 = concatenate([up4,x2])
        
        up4 = Conv2D(32, (1,1), padding='same', use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer= regularizers.l2(self.l2))(up4)
        up4 = BatchNormalization(axis=3, epsilon=1.001e-5)(up4)
        up4 = Activation('relu')(up4)
        up4 = self.dense_block(up4, 4, 8, 'stage_9',dropout= None)

        #Output layer
        output = Conv2D(1, (1,1), activation ='relu')(up4)

        model  = Model(img_input, output)

        if self.print_summary:
            print(model.summary())
        
        model.compile(optimizer = Adam(learning_rate = self.lr), loss = 'mse' )

        return model


