import cv2
import numpy as np
import os

BRAIN_TEMPLATE = "static/brain_template.png"

def overlay_on_brain(heatmap_path):

    if not os.path.exists(BRAIN_TEMPLATE):
        raise Exception("brain_template.png not found in static folder")

    brain = cv2.imread(BRAIN_TEMPLATE)

    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

    heatmap = cv2.resize(heatmap, (brain.shape[1], brain.shape[0]))

    # neural activation renk haritası
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(brain, 0.7, heatmap_color, 0.5, 0)

    output_path = "static/result.png"

    cv2.imwrite(output_path, result)

    return output_path
