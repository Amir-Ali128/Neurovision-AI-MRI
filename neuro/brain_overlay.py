import cv2
import numpy as np
import os

BRAIN_TEMPLATE = "static/brain_template.png"

def overlay_on_brain(heatmap_path):

    if not os.path.exists(BRAIN_TEMPLATE):
        raise Exception("brain_template.png not found in static folder")

    brain = cv2.imread(BRAIN_TEMPLATE)

    heatmap = cv2.imread(heatmap_path)

    heatmap = cv2.resize(heatmap, (brain.shape[1], brain.shape[0]))

    result = cv2.addWeighted(brain, 0.65, heatmap, 0.35, 0)

    output_path = "static/result.png"

    cv2.imwrite(output_path, result)

    return output_path
