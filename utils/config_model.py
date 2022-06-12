from pathlib import Path

# Deep learning libs
import torch

# Models
from models.MobileNet import *
from models.ResNet import *
from models.VGGNet import *


def reload_model(net, path: str):
    """ Reload model from path. """
    net.load_state_dict(torch.load(path))

def load_model(path: str):
    """ Load model from path. """
    return torch.load(path)

def save_checkpoints(net, epoch: int, dir_checkpoint: str, run_name: str):
    """ Save checkpoints to path. """
    Path(dir_checkpoint, run_name).mkdir(parents = True, exist_ok = True)
    # Save state dict
    torch.save(net.state_dict(), str(f"{dir_checkpoint}/{run_name}/checkpoint_epoch{epoch + 1}.pth"))
    # Save whole model
    save_model(net, dir_checkpoint, run_name)

def save_model(net, dir_checkpoint: str, run_name: str):
    """ Save model to path. """
    Path(dir_checkpoint, run_name).mkdir(parents = True, exist_ok = True)
    # Save whole model
    torch.save(net, str(f"{dir_checkpoint}/{run_name}/checkpoint_model.pth"))
    return str(f"{dir_checkpoint}/{run_name}/checkpoint_model.pth")

def conflict(args, num_classes):
    if args.output_type == 1:
        assert num_classes > 1, "Output size must be match with number of classes."
        assert args.loss_type in ['bce', 'bce_wl'], "Loss type must be 'bce' or 'bce_wl' for classification."
    else:
        assert num_classes == 1, "Output size must be 1 for one output in the end."
        assert args.loss_type in ['mse', 'mae'], "Loss type must be mse or mae for regression."

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

    elif args.model == 'VGGNet11':
        return VGGNet11(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'VGGNet13':
        return VGGNet13(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'VGGNet16':
        return VGGNet16(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    elif args.model == 'VGGNet19':
        return VGGNet19(pretrained = args.pretrained == 'True', 
                    image_channels = image_channels, num_classes = num_classes, 
                    input_size = args.input_size, **kwargs)

    else:
        assert None, 'Model not supported.'
