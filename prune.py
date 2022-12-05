import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile

# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
def prune(model, x_train, y_train, x_test, y_test, prune_epochs=2, epochs=3, batch_size=128, validation_split=0.1):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * prune_epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )

    model_for_pruning.summary()
    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(
                x_train, y_train,
                batch_size=batch_size, epochs=epochs, 
                validation_split=validation_split,
                callbacks=callbacks
    )
    
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    return model_for_pruning
