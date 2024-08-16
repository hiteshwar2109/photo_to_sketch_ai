import cv2
import numpy as np

def convert_to_sketch(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Define the default size for the image
    default_width = 800
    default_height = 600
    
    # Resize the input image to the default size while maintaining aspect ratio
    resized_img = resize_image(img, width=default_width, height=default_height)

    # Convert the resized image to grayscale
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image

    # Apply Gaussian blur to the inverted grayscale image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred_img = 255 - blurred_img

    # Divide the grayscale image by the inverted blurred image to get the sketch
    pencil_sketch = cv2.divide(gray_image, inverted_blurred_img, scale=256.0)

    # Increase the contrast of the sketch
    pencil_sketch = cv2.convertScaleAbs(pencil_sketch, alpha=1.5, beta=0)

    # Apply edge detection to sharpen the sketch
    edges = cv2.Canny(pencil_sketch, threshold1=50, threshold2=150)

    # Combine the pencil sketch with the edges to enhance sharpness
    sketch_with_edges = cv2.addWeighted(pencil_sketch, 0.8, edges, 0.2, 0)

    # Resize the sketch back to the original image dimensions
    resized_sketch = resize_image(sketch_with_edges, width=img.shape[1], height=img.shape[0])

    return resized_sketch

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # Check to see if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # Otherwise, the height is None
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # Return the resized image
    return resized
