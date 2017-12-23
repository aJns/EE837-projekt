from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Reshape
from keras.applications import vgg16

model_vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(448, 448, 3))
# model_vgg.summary(line_length=150)

# TODO: should freeze all the layers of VGG
x = MaxPooling2D(name="my_block_pool_1")(model_vgg.output)
x = Conv2D(filters=1024, kernel_size=(3,3), padding="same", name="my_block_conv_1")(x)
x = Activation("relu", name="my_block_act_1")(x)
x = Conv2D(filters=1024, kernel_size=(3,3), padding="same", name="my_block_conv_2")(x)
x = Activation("relu", name="my_block_act_2")(x)
x = Dropout(0.25, name="my_block_drop_1")(x)

x = Flatten(name="my_block_flat_1")(x)
x = Dense(4096, name="my_block_dense_1")(x)
x = Activation("relu", name="my_block_act_3")(x)
x = Dense(1470, name="my_block_dense_2")(x)
my_output = x
my_output = Reshape((7,7,30), name="my_block_reshape")(x)

# flatten = Flatten()
# new_layer2 = Dense(10, activation='softmax', name='my_dense_2')


inp = model_vgg.input
out = my_output

model_final = Model(inp, out)
model_final.summary(line_length=150)
