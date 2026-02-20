
def map_to_brain_region(neuron_id):

    if neuron_id < 160:
        return "occipital_lobe"

    elif neuron_id < 320:
        return "temporal_lobe"

    elif neuron_id < 480:
        return "parietal_lobe"

    elif neuron_id < 640:
        return "frontal_lobe"

    else:
        return "cerebellum"
