# Deep learning libs
import torch

# Models
from models.MobileNet import MobileNet_V2, MobileNet_V3
from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def reload_model(net, path: str):
    """ Reload model from path. """
    net.load_state_dict(torch.load(path))

def conflict(args, num_classes):
    if args.output_type == 1:
        assert num_classes > 1, "Output size must be match with number of classes."
    else:
        assert num_classes == 1, "Output size must be 1 for one output in the end."

def select_model(args, image_channels: int, num_classes: int, **kwargs):
    """ Select model from list of available models.

    Args:
        args (argparse): argparse object, contains the arguments of project.
        image_channels (int): number of model's image channels.
        num_classes (int): number of model's classes.

    Returns:
        Torch Model: Torch Model object.
    """
    conflict(args, num_classes)
    if args.model == 'ResNet18':
        return ResNet18(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'ResNet34':
        return ResNet34(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'ResNet50':
        return ResNet50(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'ResNet101':
        return ResNet101(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'ResNet152':
        return ResNet152(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'MobileNet_V2':
        return MobileNet_V2(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'MobileNet_V3':
        return MobileNet_V3(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    else:
        assert None, 'Model not supported.'
