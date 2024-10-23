from PIL import Image
import base64
import io
import os


# this funtion is to convert base64 in byte array
def stringToImage(base64_string):
    format, img_str = base64_string.split(';base64,')
    ext = format.split('/')[-1]
    imgdata = base64.b64decode(img_str)

    return Image.open(io.BytesIO(imgdata))


def save_image_to_folder(image_base, folder_path, new_filename=None):
    # image = Image.open(image_base)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Set the new filename if provided, otherwise use the original filename
    if new_filename:
        filename, file_extension = os.path.splitext(new_filename)

    # Generate the full path to save the image
    save_path = os.path.join(folder_path, filename + file_extension)

    # Save the image
    image_base.save(save_path)

    print(f"Image saved to {save_path}")


def image_to_base64(pil_image):
    try:
        # Convert the PIL image to bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='JPEG')
        # Encode the image data to Base64
        base64_encoded = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_encoded

    except:
        print("Error converting the image to Base64.")
        return None
