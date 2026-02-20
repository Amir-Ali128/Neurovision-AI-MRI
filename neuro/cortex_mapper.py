# neuro/cortex_mapper.py

# brain_template.png için anatomik koordinatlar
# (bu koordinatlar lateral brain template'e göre ayarlanmıştır)

BRAIN_COORDS = {

    "frontal_lobe": (180, 140),

    "parietal_lobe": (260, 120),

    "temporal_lobe": (240, 240),

    "occipital_lobe": (340, 160),

    "cerebellum": (360, 260)
}


def map_to_brain_region(neuron_id):

    # neuron id → gerçek anatomik lob

    if neuron_id < 150:
        return "occipital_lobe"

    elif neuron_id < 300:
        return "temporal_lobe"

    elif neuron_id < 450:
        return "parietal_lobe"

    elif neuron_id < 600:
        return "frontal_lobe"

    else:
        return "cerebellum"


def get_coordinates(region):

    return BRAIN_COORDS.get(region, (260,180))
