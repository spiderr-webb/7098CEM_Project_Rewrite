import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import build_models_fixed, build_pbk_meraouche_et_al, build_pbk_layer_param_change, build_optimizers, calc_l1_loss

N = 64
batch_size = 256
epochs = 50
steps_per_epoch = 300
learning_rate = 0.0008

loss_threshold = 1e-12


alice, bob, eve, pvk_gen = build_models_fixed(N)
pbk_gen = build_pbk_layer_param_change(N)

alice_optimizer, bob_optimizer, eve_optimizer, pbk_optimizer, pvk_optimizer = build_optimizers(learning_rate)


# ============================================================
# GPU-FRIENDLY DATA GENERATION
# ============================================================

@tf.function
def generate_batch():

    # Binary random tensors directly on GPU
    rand_noise = tf.random.uniform(
        (batch_size, N),
        minval=0,
        maxval=2,
        dtype=tf.int32
    )

    plaintext = tf.random.uniform(
        (batch_size, N),
        minval=0,
        maxval=2,
        dtype=tf.int32
    )

    rand_noise = tf.cast(rand_noise, tf.float32)
    plaintext = tf.cast(plaintext, tf.float32)

    return rand_noise, plaintext

# ============================================================
# TRAIN STEP
# ============================================================

@tf.function(jit_compile=True)
def train_step():

    rand_noise, plaintext = generate_batch()

    with tf.GradientTape(persistent=True) as tape:

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

        # eve_loss_detached = tf.stop_gradient(eve_loss)
        # alice_loss = bob_loss + ((eve_loss_detached - 0.5) ** 2)

        # alice_loss = bob_loss - (1 - (eve_loss ** 2))

        alice_loss = bob_loss + ((eve_loss - 0.5) ** 2)

        pbk_loss = alice_loss

    alice_grad = tape.gradient(alice_loss, alice.trainable_variables)
    bob_grad = tape.gradient(bob_loss, bob.trainable_variables)
    eve_grad = tape.gradient(eve_loss, eve.trainable_variables)
    pbk_grad = tape.gradient(pbk_loss, pbk_gen.trainable_variables)
    pvk_grad = tape.gradient(pvk_loss, pvk_gen.trainable_variables)

    alice_optimizer.apply_gradients(zip(alice_grad, alice.trainable_variables))
    bob_optimizer.apply_gradients(zip(bob_grad, bob.trainable_variables))
    eve_optimizer.apply_gradients(zip(eve_grad, eve.trainable_variables))
    pbk_optimizer.apply_gradients(zip(pbk_grad, pbk_gen.trainable_variables))
    pvk_optimizer.apply_gradients(zip(pvk_grad, pvk_gen.trainable_variables))

    return alice_loss, bob_loss, eve_loss


def training():

    bob_train_loss = []
    eve_train_loss = []

    start_time = time.time()

    with tqdm(total=epochs*steps_per_epoch, desc="Training", unit="batch") as pbar:
        for epoch in range(epochs):
            for batch in range(steps_per_epoch):

                alice_loss, bob_loss, eve_loss = train_step()

                # convert only occasionally
                if batch % 20 == 0:

                    a = float(alice_loss)
                    b = float(bob_loss)
                    e = float(eve_loss)

                    bob_train_loss.append(b)
                    eve_train_loss.append(e)

                    # update progress bar
                    pbar.set_postfix({"alice_loss": alice_loss.numpy(), "bob_loss": bob_loss.numpy(), "eve_loss": eve_loss.numpy()})
                    pbar.update()

            # exit if Alice and Bob loss is below threshold
            if (bob_loss < loss_threshold) and (0.49 < eve_loss < 0.51):
                print("Minimum loss threshold reached, exiting early")
                break

    total_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
    print(f"Training finished ({total_time})")

    # plot training errors
    plt.figure(figsize=(8, 6))
    plt.plot(bob_train_loss, label="Bob")
    plt.plot(eve_train_loss, label="Eve")
    plt.title(f"Training loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


training()
