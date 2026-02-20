import cv2
from neuro.cortex_mapper import get_coordinates

BRAIN_TEMPLATE = "static/brain_template.png"

def overlay_on_brain(activations):

    brain = cv2.imread(BRAIN_TEMPLATE)

    for act in activations:

        region = act["region"]

        x, y = get_coordinates(region)

        strength = act["activation"]

        radius = int(10 + strength * 30)

        # kırmızı neural activation
        cv2.circle(brain, (x,y), radius, (0,0,255), -1)

        # neural connection çizgisi
        cv2.line(brain, (x,y), (x+radius,y-radius), (255,255,0), 2)

    output = "static/result.png"

    cv2.imwrite(output, brain)

    return output
