# network.py
import tensorflow as tf

class UNetGenerator(tf.keras.Model):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.down1 = self.conv_block(64)
        self.down2 = self.conv_block(128)
        self.down3 = self.conv_block(256)
        self.down4 = self.conv_block(512)
        self.down5 = self.conv_block(512)
        
        # Decoder
        self.up1 = self.up_conv_block(512)
        self.up2 = self.up_conv_block(256)
        self.up3 = self.up_conv_block(128)
        self.up4 = self.up_conv_block(64)
        self.final = tf.keras.layers.Conv2D(3, kernel_size=1)

    def conv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D()
        ])

    def up_conv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Decoder avec skip connections
        u1 = self.up1(d5)
        u2 = self.up2(tf.concat([u1, d4], axis=3))
        u3 = self.up3(tf.concat([u2, d3], axis=3))
        u4 = self.up4(tf.concat([u3, d2], axis=3))

        # Upsampling final et concaténation
        d1_upsampled = tf.image.resize(d1, (256, 256))
        output = tf.image.resize(u4, (256, 256))
        
        return self.final(tf.concat([output, d1_upsampled], axis=3))

class CNNDiscriminator(tf.keras.Model):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(512, 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid'),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, x):
        return self.model(x)

