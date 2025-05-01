import tensorflow as tf

height = [170, 180, 190]
shose_size = [260, 270, 275]

# height = 170
# shose_size = 172

a = tf.Variable(0.1)
b = tf.Variable(0.5)

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)


for i in range(100):
    with tf.GradientTape() as tape:
        predicted = height * a + b
        loss = (predicted - shose_size)**2
        loss = tf.reduce_mean(loss)

    gradient = tape.gradient(loss, [a, b])
    opt.apply_gradients([[gradient[0], a], [gradient[1], b]])
    # a.assign_sub(gradient[0] * 0.00001)
    # b.assign_sub(gradient[1])
    print(a.numpy(), b.numpy())