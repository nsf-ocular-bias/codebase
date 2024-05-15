"""
Core Module for Occlusion Sensitivity
"""
import math

import numpy as np
import cv2

from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.image import apply_grey_patch
from tf_explain.utils.saver import save_rgb, save_grayscale

from warnings import warn
import tensorflow as tf

@tf.function
def transform_to_normalized_grayscale(tensor):
    """
    Transform tensor over RGB axis to grayscale.
    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, 3)
    Returns:
        tf.Tensor: 4D-Tensor of grayscale tensor, with shape (batch_size, H, W, 1)
    """
    grayscale_tensor = tf.reduce_sum(tensor, axis=-1)

    normalized_tensor = tf.cast(
        255 * tf.image.per_image_standardization(grayscale_tensor), tf.uint8
    )

    # normalize01 = tf.keras.layers.Lambda(lambda G : (G - tf.reduce_min(G))/(tf.reduce_max(G)-tf.reduce_min(G)))
    # normalized_tensor = normalize01(grayscale_tensor)

    return normalized_tensor

class OcclusionSensitivity:

    """
    Perform Occlusion Sensitivity for a given input
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(
        self,
        validation_data,
        model,
        class_index,
        patch_size,
        cls='pos',
        colormap=cv2.COLORMAP_VIRIDIS,
    ):
        """
        Compute Occlusion Sensitivity maps for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
            colormap (int): OpenCV Colormap to use for heatmap visualization

        Returns:
            np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
        """
        images, _ = validation_data
        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(model, image, class_index, patch_size, cls)
                for image in images
            ]
        )

        heatmaps = np.array(
            [
                heatmap_display(heatmap, image, colormap)
                for heatmap, image in zip(sensitivity_maps, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    def get_sensitivity_map(self, model, image, class_index, patch_size, cls):
        """
        Compute sensitivity map on a given image for a specific class index.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            image:
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image

        Returns:
            np.ndarray: Sensitivity map with shape (H, W, 3)
        """
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        patches = [
            apply_grey_patch(image, top_left_x, top_left_y, patch_size)
            for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
        ]

        coordinates = [
            (index_y, index_x)
            for index_x in range(
                sensitivity_map.shape[1]  # pylint: disable=unsubscriptable-object
            )
            for index_y in range(
                sensitivity_map.shape[0]  # pylint: disable=unsubscriptable-object
            )
        ]

        predictions = model.predict(np.array(patches), batch_size=self.batch_size)
        if cls == 'pos':
            target_class_predictions = [
                prediction[class_index] for prediction in predictions
            ]
        else:
            target_class_predictions = [
                (1 - prediction[class_index]) for prediction in predictions
            ]
        for (index_y, index_x), confidence in zip(
            coordinates, target_class_predictions
        ):
            sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)


"""
Core Module for Vanilla Gradients
"""


UNSUPPORTED_ARCHITECTURE_WARNING = (
    "Unsupported model architecture for VanillaGradients. The last two layers of "
    "the model should be: a layer which computes class scores with no activation, "
    "followed by an activation layer."
)

ACTIVATION_LAYER_CLASSES = (
    tf.keras.layers.Activation,
    tf.keras.layers.LeakyReLU,
    tf.keras.layers.PReLU,
    tf.keras.layers.ReLU,
    tf.keras.layers.Softmax,
    tf.keras.layers.ThresholdedReLU,
)


class VanillaGradients:

    """
    Perform gradients backpropagation for a given input

    Paper: [Deep Inside Convolutional Networks: Visualising Image Classification
        Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
    """

    def explain(self, validation_data, model, class_index, cls='pos'):
        """
        Perform gradients backpropagation for a given input

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect. The last two layers of
                the model should be: a layer which computes class scores with no
                activation, followed by an activation layer. This is to enable the
                gradient calculation to bypass the final activation and calculate
                the gradient of the score instead.
            class_index (int): Index of targeted class

        Returns:
            numpy.ndarray: Grid of all the gradients
        """
        score_model = self.get_score_model(model)
        return self.explain_score_model(validation_data, score_model, class_index, cls)

    def get_score_model(self, model):
        """
        Create a new model that excludes the final Softmax layer from the given model

        Args:
            model (tf.keras.Model): tf.keras model to base the new model on

        Returns:
            tf.keras.Model: A new model which excludes the last layer
        """
        activation_layer = model.layers[-1]
        if not self._is_activation_layer(activation_layer):
            warn(UNSUPPORTED_ARCHITECTURE_WARNING, stacklevel=3)
            return model

        output = activation_layer.input

        score_model = tf.keras.Model(inputs=model.inputs, outputs=output)
        return score_model

    def _is_activation_layer(self, layer):
        """
        Check whether the given layer is an activation layer

        Args:
            layer (tf.keras.layers.Layer): The layer to check

        Returns:
            Whether the layer is an activation layer
        """
        return isinstance(layer, ACTIVATION_LAYER_CLASSES)

    def explain_score_model(self, validation_data, score_model, class_index, cls):
        """
        Perform gradients backpropagation for a given input

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            score_model (tf.keras.Model): tf.keras model to inspect. The last layer
            should not have any activation function.
            class_index (int): Index of targeted class

        Returns:
            numpy.ndarray: Grid of all the gradients
        """
        images, _ = validation_data

        gradients = self.compute_gradients(images, score_model, class_index, cls)

        grayscale_gradients = transform_to_normalized_grayscale(
            tf.abs(gradients)
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    @staticmethod
    @tf.function
    def compute_gradients(images, model, class_index, cls):
        """
        Compute gradients for target class.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (batch_size, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            tf.Tensor: 4D-Tensor
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            scores = model(inputs)
            scores_for_class = scores[:, class_index]
            if cls != 'pos':
                scores_for_class = 1 - scores_for_class

        return tape.gradient(scores_for_class, inputs)

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)


"""
Core Module for SmoothGrad Algorithm
"""


class SmoothGrad:

    """
    Perform SmoothGrad algorithm for a given input

    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    """

    def explain(self, validation_data, model, class_index, num_samples=5, noise=1.0, cls='pos'):
        """
        Compute SmoothGrad for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: Grid of all the smoothed gradients
        """
        images, _ = validation_data

        noisy_images = SmoothGrad.generate_noisy_images(images, num_samples, noise)

        noisy_images = tf.convert_to_tensor(noisy_images)

        smoothed_gradients = SmoothGrad.get_averaged_gradients(
            noisy_images, model, class_index, num_samples, cls
        )

        grayscale_gradients = transform_to_normalized_grayscale(
            tf.abs(smoothed_gradients)
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        """
        Generate num_samples noisy images with std noise for each image.

        Args:
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: 4D-Tensor of noisy images with shape (batch_size*num_samples, H, W, 3)
        """
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

    @staticmethod
    @tf.function
    def get_averaged_gradients(noisy_images, model, class_index, num_samples, cls):
        """
        Compute average of gradients for target class.

        Args:
            noisy_images (tf.Tensor): 4D-Tensor of noisy images with shape
                (batch_size*num_samples, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image

        Returns:
            tf.Tensor: 4D-Tensor with smoothed gradients, with shape (batch_size, H, W, 1)
        """
        num_classes = model.output.shape[1]

        expected_output = tf.one_hot(
            [class_index] * noisy_images.shape[0],
            num_classes,
            on_value=None,
            off_value=None,
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(noisy_images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            if cls != 'pos':
                predictions = 1 - predictions
            loss = tf.keras.losses.categorical_crossentropy(
                expected_output, predictions
            )

        grads = tape.gradient(loss, inputs)

        grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
        averaged_grads = tf.reduce_mean(grads_per_image, axis=1)

        return averaged_grads

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)


"""
Core Module for Integrated Gradients Algorithm
"""

class IntegratedGradients:

    """
    Perform Integrated Gradients algorithm for a given input

    Paper: [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365.pdf)
    """

    def explain(self, validation_data, model, class_index, n_steps=10, cls='pos'):
        """
        Compute Integrated Gradients for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            np.ndarray: Grid of all the integrated gradients
        """
        images, _ = validation_data

        interpolated_images = IntegratedGradients.generate_interpolations(
            np.array(images), n_steps
        )

        interpolated_images = tf.convert_to_tensor(interpolated_images)

        integrated_gradients = IntegratedGradients.get_integrated_gradients(
            interpolated_images, model, class_index, n_steps, cls
        )

        grayscale_integrated_gradients = transform_to_normalized_grayscale(
            tf.abs(integrated_gradients)
        ).numpy()

        grid = grid_display(grayscale_integrated_gradients)

        return grid

    @staticmethod
    @tf.function
    def get_integrated_gradients(interpolated_images, model, class_index, n_steps, cls):
        """
        Perform backpropagation to compute integrated gradients.

        Args:
            interpolated_images (numpy.ndarray): 4D-Tensor of shape (N * n_steps, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            tf.Tensor: 4D-Tensor of shape (N, H, W, 3) with integrated gradients
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(interpolated_images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = predictions[:, class_index]
            if cls != 'pos':
                loss = 1 - loss

        grads = tape.gradient(loss, inputs)
        grads_per_image = tf.reshape(grads, (-1, n_steps, *grads.shape[1:]))

        integrated_gradients = tf.reduce_mean(grads_per_image, axis=1)

        return integrated_gradients

    @staticmethod
    def generate_interpolations(images, n_steps):
        """
        Generate interpolation paths for batch of images.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (N, H, W, 3)
            n_steps (int): Number of steps in the path

        Returns:
            numpy.ndarray: Interpolation paths for each image with shape (N * n_steps, H, W, 3)
        """
        baseline = np.zeros(images.shape[1:])

        return np.concatenate(
            [
                IntegratedGradients.generate_linear_path(baseline, image, n_steps)
                for image in images
            ]
        )

    @staticmethod
    def generate_linear_path(baseline, target, n_steps):
        """
        Generate the interpolation path between the baseline image and the target image.

        Args:
            baseline (numpy.ndarray): Reference image
            target (numpy.ndarray): Target image
            n_steps (int): Number of steps in the path

        Returns:
            List(np.ndarray): List of images for each step
        """
        return [
            baseline + (target - baseline) * index / (n_steps - 1)
            for index in range(n_steps)
        ]

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)



class GradientsInputs(VanillaGradients):

    """
    Perform Gradients*Inputs algorithm (gradients ponderated by the input values).
    """

    @staticmethod
    @tf.function
    def compute_gradients(images, model, class_index, cls='pos'):
        """
        Compute gradients ponderated by input values for target class.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (batch_size, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            tf.Tensor: 4D-Tensor
        """
        gradients = VanillaGradients.compute_gradients(images, model, class_index, cls)
        inputs = tf.cast(images, tf.float32)

        return tf.multiply(inputs, gradients)



class GradCAM:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(
        self,
        validation_data,
        model,
        class_index,
        layer_name=None,
        use_guided_grads=True,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
        cls='pos'
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        images, _ = validation_data

        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer(model)

        outputs, grads = GradCAM.get_gradients_and_filters(
            model, images, layer_name, class_index, use_guided_grads, cls
        )

        cams = GradCAM.generate_ponderated_output(outputs, grads)

        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    @staticmethod
    def infer_grad_cam_target_layer(model):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    @staticmethod
    def get_gradients_and_filters(
        model, images, layer_name, class_index, use_guided_grads, cls
    ):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]
            if cls != 'pos':
                loss = 1 - loss

        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = (
                tf.cast(conv_outputs > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
            )

        return conv_outputs, grads

    @staticmethod
    def generate_ponderated_output(outputs, grads):
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)

        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """

        maps = [
            GradCAM.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]

        return maps

    @staticmethod
    def ponderate_output(output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.

        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)

        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """
        weights = tf.reduce_mean(grad, axis=(0, 1))

        # Perform ponderated sum : w_i * output[:, :, i]
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

        return cam

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)
