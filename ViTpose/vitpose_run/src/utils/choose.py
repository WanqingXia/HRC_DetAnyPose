"""
choose.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script contains functions for selecting the best prediction from multiple object detection results.
It applies the mask to the original image, crops the image using the bounding box,
and resizes the image to the required input size for the DINOv2 model.
The script then calculates the cosine similarity between the image embedding
and the reference embeddings for each viewpoint, and selects the prediction with the
highest average cosine similarity.
"""

import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from utils.similarity import CosineSimilarity
from utils.convert import Convert_YCB

def draw_outcome(img, pred, best_pred):
    """
    Draw the prediction outcome on the image using PIL.
    :param img: RGB cv2 image
    :param pred: the prediction of masks which includes 'boxes', 'scores', 'labels', 'masks'
    :param best_pred: index of the best prediction
    """
    # Convert cv2 image (BGR) to PIL image (RGB)
    img = Image.fromarray(img)

    # Draw the mask
    mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
    mask = np.squeeze(mask, axis=0)  # Remove single channel dimension if present
    mask_img = Image.fromarray(mask * 255, mode='L')

    # Create a black image
    black_img = Image.new("RGB", img.size, (0, 0, 0))

    # Composite the original image and the black image using the mask
    img = Image.composite(img, black_img, mask_img)

    draw = ImageDraw.Draw(img)

    # Draw the bounding box
    box = pred['boxes'][best_pred].cpu().numpy().astype(int)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=2)

    # Draw the label
    label = str(pred['labels'][best_pred])
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_location = (box[0], box[1] - (text_bbox[3] - text_bbox[1]))
    draw.text(text_location, label, fill=(0, 255, 0), font=font)

    # Convert the image to a format that Matplotlib can display
    img_array = np.array(img)

    # Display the image with Matplotlib and set the title
    plt.imshow(img_array)
    plt.title('Dinov2 Selection Result')
    plt.axis('off')  # Hide the axis
    plt.show()


def get_embedding(image, mask, box, dinov2):
    # Apply the mask directly to the original image
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Crop the masked image using the bounding box
    x0, y0, x1, y1 = map(int, box)
    cropped_masked_img = masked_img[y0:y1, x0:x1]

    # Calculate the size needed for a square image
    height, width, _ = cropped_masked_img.shape
    side_length = max(width, height)
    square_img = np.zeros((side_length, side_length, 3), dtype=np.uint8)

    # Calculate the position to place the cropped image in the square image
    x_offset = (side_length - width) // 2
    y_offset = (side_length - height) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_masked_img

    # Resize the square image to dinov2's required input size
    final_img_resized = cv2.resize(square_img, dinov2.dinov2_size, interpolation=cv2.INTER_LANCZOS4)

    # Forward through DinoV2 and get embedding
    img_array = np.array(final_img_resized)
    embed_img = dinov2.forward(img_array)
    embed_img = embed_img.detach().cpu()  # Detach from GPU
    return embed_img, final_img_resized


def validate_preds(img, pred, dinov2, show_result=False):
    """
    :param img: RGB cv2 image
    :param pred: the prediction of masks which includes 'boxes', 'scores', 'labels', 'masks'
    :param dinov2: DinoV2 model, used to embed the image to a tensor of (1,1536)
    :return: best_pred: index of the best prediction
    """
    num_predictions = len(pred['labels'])
    CosineSim = CosineSimilarity()
    convert_YCB = Convert_YCB()
    best_pred = 0
    if num_predictions > 1:
        original_label = pred['labels'][0]
        label = convert_YCB.convert_name(original_label)
        cos_similarities = np.zeros((num_predictions, len(dinov2.viewpoints_embeddings[label])))

        # Process each prediction mask
        for i in range(num_predictions):
            img_copy = img.copy()  # preserve the original image
            mask = pred['masks'][i].cpu().numpy().astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))  # Change order to (H, W, C) for CV2
            box = pred['boxes'][i].cpu().numpy().astype(int)  # Format: [x0, y0, x1, y1]
            embed_img, _ = get_embedding(img_copy, mask, box, dinov2)

            # Calculate similarity
            reference_embedding = dinov2.viewpoints_embeddings[label]
            cos_similarities[i, :] = CosineSim(embed_img, reference_embedding)

        # Choose the best viewpoint
        best_pred = np.argmax(np.mean(cos_similarities, axis=1))

    if show_result:
        draw_outcome(img.copy(), pred, best_pred)
    return best_pred
