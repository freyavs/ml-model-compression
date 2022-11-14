import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile

from distiller import Distiller


#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_mnist_model():
    # Define the model architecture.
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])
    
    return model

def get_teacher():
    # Create the teacher
    teacher = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="teacher",
    )

    return teacher

def get_student():
    # Create the student
    student = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="student",
    )

    return student

# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
def prune(model, x_train, y_train, x_test, y_test, epochs=2, batch_size=128, validation_split=0.1):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    num_images = x_train.shape[0] # * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model_for_pruning.summary()
    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs, validation_split=0,
                    callbacks=callbacks)
    
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
   x_test, y_test, verbose=0)

    print('Pruned test accuracy:', model_for_pruning_accuracy)
    return model_for_pruning

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

def compression_result(model_before_optimization, model_optimized):
    # calculate MB
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_optimized)

    _, optimized_keras_file = tempfile.mkstemp('.h5')
    _, baseline_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, optimized_keras_file, include_optimizer=False)
    tf.keras.models.save_model(model_before_optimization, baseline_keras_file, include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print("---- MB optimization ----")
    print("Size of gzipped baseline Keras model: %.2f KB" % (get_gzipped_model_size(baseline_keras_file)* 0.001))
    print("Size of gzipped optimized Keras model: %.2f KB" % (get_gzipped_model_size(optimized_keras_file)* 0.001))

    # calculate parameter difference
    print("---- parameter optimization ----")
    print("Baseline model parameters: %.0f" % model_before_optimization.count_params())
    print("Optimized model parameters: %.0f" % model_optimized.count_params())
    return


def main():
    teacher = get_teacher()
    student = get_student()

    # Clone student for later comparison
    student_scratch = keras.models.clone_model(student)

    # Prepare the train and test dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    # Train teacher as usual
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate teacher on data.
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model('teacher')
        teacher.evaluate(x_test, y_test)
    else:
        teacher.fit(x_train, y_train, epochs=5)
        teacher.evaluate(x_test, y_test)
        teacher.save('teacher')

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    distiller.fit(x_train, y_train, epochs=3)

    # Evaluate student on test dataset
    distiller.evaluate(x_test, y_test)
    teacher.save('student')

    # Train student as doen usually
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate student trained from scratch.
    student_scratch.fit(x_train, y_train, epochs=3)
    student_scratch.evaluate(x_test, y_test)
    teacher.save('student_scratch')

    compression_result(teacher, student)

if __name__ == '__main__':
    main()
