__author__ = 'mkv-aql'
from tensorflow import keras
#from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Maxpool2D, Conv2DTranspose, Concatenate, Input
#from tensorflow.keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model

## Convolution (encoder and decoder bridge)
def convolution_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

## Encoder
def encoder_block(input, num_filters):
    x = convolution_block(input, num_filters)
    p = MaxPool2D((2, 2))(x) #handles the input for the 2nd block

    return x, p

## Decoder
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input) #pixel input is 299x299, it will be doubled to 598x598
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, num_filters)

    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64) # s1 skip conncetion, and p1 (pulling) is the output, takes input images with 64 filters
    s2, p2 = encoder_block(p1, 128) #Increase by 2x
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = convolution_block(p4, 1024) #bridge

    d1 = decoder_block(b1, s4, 512) #decoder
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()



