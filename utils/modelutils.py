
def count_model_params(model):
    """
    Returns number of trainable and non-trainable parameters in a model.
    :param model: A PyTorch nn.Module object.
    :return: A tuple (train_params_count, non_train_params_count)
    """

    train_params_count = 0
    non_train_params_count = 0
    for p in model.parameters():
        if p.requires_grad:
            train_params_count += p.numel()
        else:
            non_train_params_count += p.numel()

    return train_params_count, non_train_params_count