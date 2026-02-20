# neuro/cortex_mapper.py

# Anatomical coordinates for brain_template.png
BRAIN_REGIONS = {
    "Frontal_lobe": {
        "coord": (250, 180),
        "color": (255, 0, 0)
    },
    "Parietal_lobe": {
        "coord": (320, 140),
        "color": (0, 255, 0)
    },
    "Temporal_lobe": {
        "coord": (280, 240),
        "color": (0, 0, 255)
    },
    "Occipital_lobe": {
        "coord": (380, 200),
        "color": (255, 255, 0)
    },
    "Cerebellum": {
        "coord": (400, 280),
        "color": (255, 0, 255)
    }
}


def map_to_brain_region(neuron_id):

    if neuron_id < 150:
        region = "Occipital_lobe"

    elif neuron_id < 300:
        region = "Temporal_lobe"

    elif neuron_id < 450:
        region = "Parietal_lobe"

    elif neuron_id < 600:
        region = "Frontal_lobe"

    else:
        region = "Cerebellum"

    return {
        "region": region,
        "coord": BRAIN_REGIONS[region]["coord"],
        "color": BRAIN_REGIONS[region]["color"]
    }
