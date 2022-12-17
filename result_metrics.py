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

def compression_result(model, name:str="dummy", pruned = False, save=lambda f,a: (f,a)):
    # calculate MB

    model_for_export = model
    if pruned:
      model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    _, keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, keras_file, include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    tflite_model = converter.convert()

    _, tflite_file = tempfile.mkstemp('.tflite')

    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)

    print("---- Compression result ----")
    print(f"Size of gzipped {name} model: {get_gzipped_model_size(keras_file)* 0.001}KB")
    save(f"{name}_size", (get_gzipped_model_size(keras_file)* 0.001))

    # calculate parameter difference
    print(f"Baseline model {name} parameters: {model.count_params()}")
    save(f"{name}_parameters", (model.count_params()))
    return
