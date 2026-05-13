import string

import tensorflow as tf
import numpy as np
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


def build_models():

    alice, bob, eve, pvk_gen = build_models_fixed(N)
    pbk_gen = build_pbk_layer_param_change(N)

    return alice, bob, eve, pbk_gen, pvk_gen


def generate_noise_batch(b_size, datasize):

    # bin(random.getrandbits(N))[2:].zfill(N)

    return np.random.choice([0, 1], size=(b_size, datasize))


def generate_msg_batch(b_size, datasize):

    # s = ''.join(random.choice(string.printable) for _ in range(datasize//8))
    # a = [bin(ord(c))[2:].zfill(8) for c in s]

    #printable_arr = [bin(ord(c))[2:].zfill(8) for c in string.printable]
    #printable_arr = [bord(c))[2:].zfill(8) for c in string.printable]

    #arr = np.array([list(bin(ord(c))[2:].zfill(8)) for c in ''.join(random.choice(string.printable) for _ in range(datasize//8))]).astype(int)

    # np.random.choice(list(string.printable), size=(b_size, datasize//8))

    str_arr = np.array([''.join(row) for row in np.random.choice(list(string.printable), size=(b_size, datasize//8))])

    bin_arr = []
    for s in str_arr:
        bin_arr.append(np.array([list(bin(ord(c))[2:].zfill(8)) for c in s]).astype(int).ravel())

    return np.array(bin_arr)


def training(alice, bob, eve, pbk_gen, pvk_gen):

    bob_train_loss = []
    eve_train_loss = []

    start_time = time.time()

    with tqdm(total=epochs*steps_per_epoch, desc="Training", unit="batch") as pbar:
        for epoch in range(epochs):
            for batch in range(steps_per_epoch):

                rand_noise = generate_noise_batch(batch_size, N)
                plaintext = generate_msg_batch(batch_size, N)

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

                    bob_train_loss.append(bob_loss)
                    eve_train_loss.append(eve_loss)

                    pvk_loss = bob_loss

                    # eve_loss_detached = tf.stop_gradient(eve_loss)
                    # alice_loss = bob_loss + ((eve_loss_detached - 0.5) ** 2)

                    # alice_loss = bob_loss - (1 - (eve_loss ** 2))

                    alice_loss = bob_loss + ((eve_loss - 0.5) ** 2)

                    pbk_loss = alice_loss

                    # print("\n\nAlice / PBK: ")
                    # print(alice_loss)
                    # print("\n\nBob / PVK: ")
                    # print(bob_loss)
                    # print("\n\nEve: ")
                    # print(eve_loss)
                    # print("\n\n")

                    if epoch+batch == 0:
                        alice.summary()
                        bob.summary()
                        eve.summary()
                        pbk_gen.summary()
                        pvk_gen.summary()

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
    # plt.yticks(np.arange(0, (N * 0.75), 1))
    plt.legend()
    plt.show()

    return alice, bob, eve, pbk_gen, pvk_gen


def evaluation(alice, bob, eve, pbk_gen, pvk_gen):

    bob_eval_loss = []
    eve_eval_loss = []

    start_time = time.time()

    with tqdm(total=steps_per_epoch, desc="Evaluation", unit="batch") as pbar:
        for batch in range(steps_per_epoch):

            rand_noise = generate_noise_batch(batch_size, N)
            plaintext = generate_msg_batch(batch_size, N)

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

                bob_eval_loss.append(bob_loss)
                eve_eval_loss.append(eve_loss)

                pvk_loss = bob_loss

                # alice_loss = bob_loss + (1 - (eve_loss ** 2))

                alice_loss = bob_loss + ((eve_loss - 0.5) ** 2)

                pbk_loss = alice_loss

                # update progress bar
                pbar.set_postfix({"alice_loss": alice_loss.numpy(), "bob_loss": bob_loss.numpy(), "eve_loss": eve_loss.numpy()})
                pbar.update()

    total_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
    print(f"Evaluation finished ({total_time})")

    # plot training errors
    plt.figure(figsize=(8, 6))
    plt.plot(bob_eval_loss, label="Bob")
    plt.plot(eve_eval_loss, label="Eve")
    plt.title(f"Evaluation loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    # plt.yticks(np.arange(0, (N * 0.75), 1))
    plt.legend()
    plt.show()


new_alice, new_bob, new_eve, new_pbk, new_pvk = build_models()
alice_optimizer, bob_optimizer, eve_optimizer, pbk_optimizer, pvk_optimizer = build_optimizers(learning_rate)

trained_alice, trained_bob, trained_eve, trained_pbk, trained_pvk = training(new_alice, new_bob, new_eve, new_pbk, new_pvk)
evaluation(trained_alice, trained_bob, trained_eve, trained_pbk, trained_pvk)
