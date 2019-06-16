import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('acc') > 0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
    

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255


model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks=myCallback()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
model.evaluate(x_test, y_test)
classify = model.predict(x_test)
print(classify[0])
print(y_test[0])
