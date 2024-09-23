import os
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.DEBUG)

saved_model_path = 'saved_model/model.keras'

# Check if the Keras model file exists
if not os.path.exists(saved_model_path):
    print(f"Model file does not exist at: {saved_model_path}")
else:
    # Load the Keras model
    model = tf.keras.models.load_model(saved_model_path)

    # Convert the Keras model to a TensorFlow function with a fixed input shape
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Initialize the TFLite converter using the TensorFlow function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([full_model])

    # Apply optimizations for size and compatibility
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops as fallback for unsupported TFLite ops.
    ]

    # Convert the model
    tflite_model = converter.convert()

    # Save the converted TFLite model
    tflite_model_path = 'saved_model/model_quantized.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Conversion to TensorFlow Lite model completed successfully. Quantized model saved at: {tflite_model_path}")
