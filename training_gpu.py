import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import build_models_fixed, build_pbk_meraouche_et_al, build_pbk_layer_param_change, build_optimizers, calc_l1_loss


N = 64
batch_size = 256
epochs = 200
steps_per_epoch = 300
learning_rate = 0.0008

loss_threshold = 1e-12

# tf.keras.mixed_precision.set_global_policy('mixed_float16')


alice, bob, eve, pvk_gen = build_models_fixed(N)
pbk_gen = build_pbk_v3(N)

alice_optimizer, bob_optimizer, eve_optimizer, pbk_optimizer, pvk_optimizer = build_optimizers(learning_rate)


@tf.function
def generate_batch():

  dtype = tf.keras.mixed_precision.global_policy().compute_dtype

  rand_noise = (2 * tf.random.uniform((batch_size, N), minval=0, maxval=2, dtype=tf.int32)) - 1

  plaintext = (2 * tf.random.uniform((batch_size, N), minval=0, maxval=2, dtype=tf.int32)) - 1

  rand_noise = tf.cast(rand_noise, dtype)
  plaintext = tf.cast(plaintext, dtype)

  return rand_noise, plaintext


@tf.function(jit_compile=True)
def train_alice_bob():

    rand_noise, plaintext = generate_batch()

    with tf.GradientTape(persistent=True) as tape:

        eve.trainable = False

        pub_key = pbk_gen(rand_noise)

        pvk_in = tf.concat([rand_noise, pub_key], axis=1)
        priv_key = pvk_gen(pvk_in)

        alice_in = tf.concat([plaintext, pub_key], axis=1)
        ciphertext = alice(alice_in)

        bob_in = tf.concat([ciphertext, priv_key], axis=1)
        plaintext_b = bob(bob_in)

        eve_in = tf.concat([ciphertext, pub_key], axis=1)
        plaintext_e = eve(eve_in)

        bob_loss = calc_l1_loss(plaintext, plaintext_b)
        eve_loss = calc_l1_loss(plaintext, plaintext_e)

        pvk_loss = bob_loss

        # alice_loss = bob_loss + (1 - (eve_loss ** 2)) # meraouche loss function

        alice_loss = bob_loss + (((0.5 - eve_loss) ** 2) / 0.25) # abadi and andersen loss function

        pbk_loss = alice_loss

    alice_grad = tape.gradient(alice_loss, alice.trainable_variables)
    bob_grad = tape.gradient(bob_loss, bob.trainable_variables)
    pbk_grad = tape.gradient(pbk_loss, pbk_gen.trainable_variables)
    pvk_grad = tape.gradient(pvk_loss, pvk_gen.trainable_variables)

    alice_optimizer.apply_gradients(zip(alice_grad, alice.trainable_variables))
    bob_optimizer.apply_gradients(zip(bob_grad, bob.trainable_variables))
    pbk_optimizer.apply_gradients(zip(pbk_grad, pbk_gen.trainable_variables))
    pvk_optimizer.apply_gradients(zip(pvk_grad, pvk_gen.trainable_variables))

    del tape

    return alice_loss, bob_loss, eve_loss # , test1, test2


@tf.function(jit_compile=True)
def train_eve():

    rand_noise, plaintext = generate_batch()

    with tf.GradientTape() as tape:

        eve.trainable = True

        pub_key = pbk_gen(rand_noise)

        pvk_in = tf.concat([rand_noise, pub_key], axis=1)
        priv_key = pvk_gen(pvk_in)

        alice_in = tf.concat([plaintext, pub_key], axis=1)
        ciphertext = alice(alice_in)

        eve_in = tf.concat([ciphertext, pub_key], axis=1)
        plaintext_e = eve(eve_in)

        eve_loss = calc_l1_loss(plaintext, plaintext_e)

    eve_grad = tape.gradient(eve_loss, eve.trainable_variables)

    eve_optimizer.apply_gradients(zip(eve_grad, eve.trainable_variables))

    return eve_loss


def training():

    bob_train_loss = []
    eve_train_loss = []

    start_time = time.time()

    with tqdm(total=epochs*steps_per_epoch, desc="Training", unit="batch") as pbar:
        for epoch in range(epochs):
            for batch in range(steps_per_epoch):

              if (batch % 6 == 0):
                
                alice_loss, bob_loss, eve_loss = train_alice_bob()

              else:

                eve_loss = train_eve()


              a = float(alice_loss)
              b = float(bob_loss)
              e = float(eve_loss)

              # update progress bar
              pbar.set_postfix({"alice_loss": alice_loss.numpy(), "bob_loss": bob_loss.numpy(), "eve_loss": eve_loss.numpy()})
              pbar.update()

            bob_train_loss.append(b)
            eve_train_loss.append(e)

            # exit if Alice and Bob loss is below threshold
            if (bob_loss < loss_threshold) and (0.47 < eve_loss < 0.53):
                print("Minimum loss threshold reached, exiting early")
                break

    total_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
    print(f"Training finished ({total_time})")

    # plot training errors
    plt.figure(figsize=(8, 6))
    plt.plot(bob_train_loss, label="Bob")
    plt.plot(eve_train_loss, label="Eve")
    plt.title(f"Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.legend()
    plt.show()


training()
