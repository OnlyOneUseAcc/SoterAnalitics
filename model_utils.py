from tensorflow import keras as ks


def build_model(input_shape, n_classes):
    x_input = ks.Input(shape=(input_shape, 1))
    y_input = ks.Input(shape=(input_shape, 1))
    z_input = ks.Input(shape=(input_shape, 1))

    x_layer = ks.layers.Conv1D(filters=64,
                               kernel_size=5,
                               padding='same')(x_input)
    x_layer = ks.layers.MaxPooling1D(pool_size=3)(x_layer)
    x_layer = ks.layers.BatchNormalization()(x_layer)
    x_layer = ks.layers.ReLU()(x_layer)
    x_layer = ks.layers.Dropout(0.4)(x_layer)

    x_layer = ks.layers.Flatten()(x_layer)
    x_layer = ks.layers.Dense(64, 'relu')(x_layer)

    y_layer = ks.layers.Conv1D(filters=64,
                               kernel_size=5,
                               padding='same')(y_input)
    y_layer = ks.layers.MaxPooling1D(pool_size=3)(y_layer)
    y_layer = ks.layers.BatchNormalization()(y_layer)
    y_layer = ks.layers.ReLU()(y_layer)
    y_layer = ks.layers.Dropout(0.4)(y_layer)

    y_layer = ks.layers.Flatten()(y_layer)
    y_layer = ks.layers.Dense(64, 'relu')(y_layer)

    z_layer = ks.layers.Conv1D(filters=64,
                               kernel_size=5,
                               padding='same')(z_input)
    z_layer = ks.layers.MaxPooling1D(pool_size=3)(z_layer)
    z_layer = ks.layers.BatchNormalization()(z_layer)
    z_layer = ks.layers.ReLU()(z_layer)
    z_layer = ks.layers.Dropout(0.4)(z_layer)

    z_layer = ks.layers.Flatten()(z_layer)
    z_layer = ks.layers.Dense(64, 'relu')(z_layer)

    x_model = ks.Model(inputs=x_input, outputs=x_layer)
    y_model = ks.Model(inputs=y_input, outputs=y_layer)
    z_model = ks.Model(inputs=z_input, outputs=z_layer)

    conc_layer = ks.layers.Concatenate()([x_model.output, y_model.output, z_model.output])
    conc_layer = ks.layers.Dense(64, activation='relu')(conc_layer)
    conc_layer = ks.layers.Dense(n_classes, activation='softmax')(conc_layer)

    return ks.Model(
        inputs=[x_model.input,
                y_model.input,
                z_model.input],
        outputs=conc_layer
    )
