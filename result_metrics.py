import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tempfile

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

def compression_result(model_before_optimization, model_optimized, name:str, save=lambda f,a: (f,a)):
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
    #save(f"{name}_size", (get_gzipped_model_size(baseline_keras_file)* 0.001, get_gzipped_model_size(optimized_keras_file)* 0.001))

    # calculate parameter difference
    print("---- parameter optimization ----")
    print("Baseline model parameters: %.0f" % model_before_optimization.count_params())
    print("Optimized model parameters: %.0f" % model_optimized.count_params())
    #save(f"{name}_parameters", (model_before_optimization.count_params(), model_optimized.count_params()))
    return
