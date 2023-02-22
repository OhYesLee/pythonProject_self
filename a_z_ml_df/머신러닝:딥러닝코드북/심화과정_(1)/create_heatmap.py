import tensorflow as tf
import numpy as np
def gradcam(image,classfication_index,model,layer_number):
    selected_layer = model.layers[layer_number]
    selected_layer_model = tf.keras.Model(model.inputs, selected_layer.output)
    classifier_input = tf.keras.Input(shape=selected_layer.output.shape[1:])
    x = classifier_input
    x =model.layers[-2](x)
    outputs = model.layers[-1](x)
    classifier_model= tf.keras.Model(inputs=classifier_input,outputs=outputs)
    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        selected_layer_output = selected_layer_model(inputs)
        tape.watch(selected_layer_output)
        preds = classifier_model(selected_layer_output)
        pred_index = classfication_index
        selected_class_channel = preds[:, pred_index]
    grads = tape.gradient(selected_class_channel, selected_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    selected_layer_output = selected_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        selected_layer_output[:, :, i] *= pooled_grads[i]
    gradcam = np.mean(selected_layer_output, axis=-1)
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    return gradcam

