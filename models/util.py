from . import model_dict


def create_model(name, n_cls, dataset='cifar100', dropout=0.1):
    if dataset == 'cifar100':
        if name == 'resnet18_3':
            model = model_dict[name](num_classes=n_cls)
        elif name == 'convnet4':
            model = model_dict[name](num_classes=n_cls)
    else:
        raise NotImplementedError('Not supported')
    return model
