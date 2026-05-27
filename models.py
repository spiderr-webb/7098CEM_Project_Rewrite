import tensorflow as tf


def define_model_1(input_size):  # for alice, bob, eve, pvk_gen, and attacker 1, also og pbk_gen

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Reshape((input_size, 1)),
        tf.keras.layers.Conv1D(filters=2, kernel_size=4, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=2, strides=2, padding='valid', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh'),
        tf.keras.layers.Flatten()
    ])

    return model


def define_model_2(input_size):  # option 1 for pbk_gen

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Reshape((input_size, 1)),
        tf.keras.layers.Conv1D(filters=2, kernel_size=4, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh'),
        tf.keras.layers.Flatten()
    ])

    return model


def define_model_3(input_size):  # option 2 for pbk_gen

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Reshape((input_size, 1)),
        tf.keras.layers.Conv1D(filters=2, kernel_size=4, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=2, strides=2, padding='valid', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(input_size, activation='softmax')
    ])

    return model


def define_model_4(input_size):  # for attackers 2 and 3 (or 3 and 4 gd it Meraouche be consistent pls), also option 3 for pbk_gen

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Reshape((input_size, 1)),
        tf.keras.layers.Conv1D(filters=2, kernel_size=4, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=2, strides=2, padding='valid', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(input_size, activation='softmax')
    ])

    return model


def build_models_fixed(N):

    alice = define_model_1(2*N)
    bob = define_model_1(2*N)
    eve = define_model_1(2*N)
    pvk_gen = define_model_1(2*N)

    return alice, bob, eve, pvk_gen


def build_pbk_meraouche_et_al(N):

    pbk_gen = define_model_1(N)
    return pbk_gen


def build_pbk_layer_param_change(N):

    pbk_gen = define_model_2(N)
    return pbk_gen


def build_pbk_add_layer_same_activate(N):

    pbk_gen = define_model_3(N)
    return pbk_gen


def build_models_pbk_add_layer_change_activate(N):

    pbk_gen = define_model_4(N)
    return pbk_gen


def build_optimizers(learning_rate):

    alice_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bob_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eve_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    pbk_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    pvk_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return alice_optimizer, bob_optimizer, eve_optimizer, pbk_optimizer, pvk_optimizer


def calc_l1_loss(p_in, p_out):

    p_in_scaled = (p_in + 1) / 2
    p_out_scaled = (p_out + 1) / 2

    return tf.reduce_mean(tf.abs(p_in_scaled - p_out_scaled))
