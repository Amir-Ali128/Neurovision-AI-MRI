import cv2
import numpy as np
import os

OUTPUT_FOLDER = "static/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def overlay_heatmap(image_path, heatmap_array, output_name="result.jpg"):
    """
    MRI görüntüsü üstüne heatmap bindirir
    """

    # original MRI image
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image could not be loaded")

    image = cv2.resize(image, (512, 512))

    # heatmap normalize
    heatmap = heatmap_array

    if len(heatmap.shape) == 3:
        heatmap = heatmap.squeeze()

    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)

    # apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # overlay
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    cv2.imwrite(output_path, overlay)

    return output_path
