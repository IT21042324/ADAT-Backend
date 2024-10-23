
import base64
import io
import warnings
# import tensorflow.compat.v2 as tf
import tensorflow as tf
from PIL import Image

try:
    from PIL import Image as pil_image

    try:
        pil_image_resampling = pil_image.Resampling
    except AttributeError:
        pil_image_resampling = pil_image
except ImportError:
    pil_image = None
    pil_image_resampling = None

if pil_image_resampling is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image_resampling.NEAREST,
        "bilinear": pil_image_resampling.BILINEAR,
        "bicubic": pil_image_resampling.BICUBIC,
        "hamming": pil_image_resampling.HAMMING,
        "box": pil_image_resampling.BOX,
        "lanczos": pil_image_resampling.LANCZOS,
    }

ResizeMethod = tf.image.ResizeMethod

_TF_INTERPOLATION_METHODS = {
    "bilinear": ResizeMethod.BILINEAR,
    "nearest": ResizeMethod.NEAREST_NEIGHBOR,
    "bicubic": ResizeMethod.BICUBIC,
    "area": ResizeMethod.AREA,
    "lanczos3": ResizeMethod.LANCZOS3,
    "lanczos5": ResizeMethod.LANCZOS5,
    "gaussian": ResizeMethod.GAUSSIAN,
    "mitchellcubic": ResizeMethod.MITCHELLCUBIC,
}


def string_to_image(base64_string: str) -> Image.Image:
    """
    input :-  image encoded as a base_64 string
    output :- Image in RGB format
    description :- Process an image encoded as a base_64 string
    """

    _, img_str = base64_string.split(";base64,")
    imgdata = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(imgdata))

    return image.convert("RGB")


def load_img(
        img,
        grayscale=False,
        color_mode="rgb",
        target_size=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
):
    """Loads an image into PIL format.
    Usage:
    ```python
    image = tf.keras.utils.load_img(image_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    ```
    Args:
        img: PIL Image.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
          The desired image format.
        target_size: Either `None` (default to original size) or tuple of ints
          `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
          1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
          version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
          supported. By default, `"nearest"` is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.
    Returns:
        A PIL Image instance.
    Raises:
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale:
        warnings.warn(
            'grayscale is deprecated. Please use color_mode = "grayscale"'
        )
        color_mode = "grayscale"

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys()),
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img
