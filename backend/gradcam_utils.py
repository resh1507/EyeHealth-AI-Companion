import numpy as np
import tensorflow as tf
import cv2
from PIL import Image



def get_gradcam_heatmap(model, img_array, last_conv_layer_name="top_conv"):
    """
    Generate a Grad-CAM heatmap for a given model and input image.

    Args:
        model: A trained Keras model.
        img_array: Preprocessed image of shape (1, H, W, 3).
        last_conv_layer_name: Name of the last conv layer in the model.

    Returns:
        heatmap: A 2D numpy array (H, W) with values in [0,1].
    """

    # Build a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward pass
    conv_outputs, predictions = grad_model(img_array)
    print("Predictions shape:", predictions.shape)  # Debug info

    # Handle classification output
    if predictions.ndim == 2:  # (batch_size, num_classes)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]
    else:  # e.g., (num_classes,)
        pred_index = tf.argmax(predictions)
        pred_output = predictions[pred_index]

    # Compute gradients of the top predicted class wrt last conv outputs
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.ndim == 2:
            pred_output = predictions[:, pred_index]
        else:
            pred_output = predictions[pred_index]

    grads = tape.gradient(pred_output, conv_outputs)

    # Pool gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv layer outputs by the pooled gradients
    conv_outputs = conv_outputs[0]  # remove batch dimension
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap)
    heatmap /= max_val

    return heatmap


def overlay_gradcam(img_input, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on image.
    
    Args:
        img_input: file path (str) or numpy array (RGB or BGR)
        heatmap: numpy array (0–1)
        alpha: blending factor
        colormap: cv2 colormap
    
    Returns:
        overlayed BGR image (numpy array)
    """
    # Load image if input is a path
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Could not read image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        # Ensure BGR for consistency
        if img.shape[-1] == 3 and img.max() <= 1.0:
            img = (img * 255).astype("uint8")
        if img.shape[-1] == 3 and img[..., 0].mean() < img[..., 2].mean():  
            # Likely RGB → convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("img_input must be a file path or numpy array")

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay