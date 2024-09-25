from .models import CSRNet, SFANet, Bay, DMCount
from .util.functions import load_image

import torch


def load_model(model_name="CSRNet", model_weights="SHA"):
    """
    Builds a model for Crowd Counting and initializes it as a singleton.
    :param model_name: One of the available models: CSRNet.
    :param model_weights: Name of the dataset the model was pretrained on. Possible values vary on the model.
    :return: Built Crowd Counting model initialized with pretrained weights.
    """

    available_models = {
        'CSRNet': CSRNet,
        'SFANet': SFANet,
        'Bay': Bay,
        'DM-Count': DMCount
    }

    global loaded_models

    if "loaded_models" not in globals():
        loaded_models = {}

    model_full_name = "{}_{}".format(model_name, model_weights)
    if model_full_name not in loaded_models.keys():
        model = available_models.get(model_name)
        if model:
            model = model.make_model(model_weights)
            loaded_models[model_full_name] = model
            print("Built model {} with weights {}".format(
                model_name, model_weights))
        else:
            raise ValueError(
                "Invalid model_name. Model {} is not available.".format(model_name))

    return loaded_models[model_full_name]


def get_count(img_paths, model_name="CSRNet", model_weights="SHA", model=None, is_gray=False, return_density=False,
              resize_img=True):
    """
    Return the count on image/s. You can use already loaded model or choose the name and pre-trained weights.
    """

    # if one path to array
    if type(img_paths) != list:
        img_paths = [img_paths]

    # load model
    if model is None:
        model = load_model(model_name, model_weights)

    # load images
    imgs, names = [], []

    for img_path in img_paths:
        img, name = load_image(img_path, model.get_name(), is_gray, resize_img)
        imgs.append(img)
        names.append(name)

    # Concatenar las imágenes y moverlas a la GPU
    # Mover las imágenes a la GPU
    imgs = torch.cat(imgs).to(torch.device('cuda'))

    with torch.set_grad_enabled(False):
        outputs = model(imgs)  # Ejecutar el modelo en la GPU

    # Mover los resultados de vuelta a la CPU para su manejo
    counts = torch.sum(outputs, (1, 2, 3)).cpu().numpy()
    counts = dict(zip(names, counts))

    # Mover densidades a la CPU
    densities = dict(zip(names, outputs[:, 0, :, :].cpu().numpy()))

    if len(counts) == 1:
        if return_density:
            return counts[name], densities[name]
        else:
            return counts[name]

    if return_density:
        return counts, densities

    return counts
