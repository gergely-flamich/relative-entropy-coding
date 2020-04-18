import time

import numpy as np
import tensorflow as tf

from rec.io.entropy_coding import ArithmeticCoder


num_symbols = 2**6
message_length = 2000
test_file_path = "scratch_compression_test.miracle"

P = np.ones(num_symbols + 1, dtype=np.int32)
P[1:] = np.random.choice(100, size=num_symbols) + 1

message = np.zeros(message_length, dtype=np.int32)

message[:-1] = np.random.choice(num_symbols, size=message_length - 1) + 1

ac = ArithmeticCoder(P, precision=32)

start = time.time()

print("Coding..")
code = ac.encode(message)
print("Coded in {:.4f}s".format(time.time() - start))

log_P = np.log(P)
log_P_normed = log_P - tf.reduce_logsumexp(tf.cast(log_P, tf.float64))
log_P_normed = log_P_normed / np.log(2)

expected_message_length = 0
for m in message:
    expected_message_length -= log_P_normed[m]

print(f"Expected message length: {expected_message_length:.2f} bits!")
print(f"Actual message length: {len(''.join(code))} bits!")

start = time.time()

print("Fast Decoding...")
decompressed = ac.decode_fast(code)
print("Decoded in {:.4f}s".format(time.time() - start))


print(np.all(decompressed == message))
