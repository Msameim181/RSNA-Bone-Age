import argparse
from xmlrpc.client import Boolean


def get_args():
    parser = argparse.ArgumentParser(description='Train the Your Model on images and target age.')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')

    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='learning_rate')

    parser.add_argument('--validation', '-v', dest='val_percent', type=float, default=0.3,
                        help='Percent of the data that is used as validation (0-1)')
    
    parser.add_argument('--model-type', '-m', dest='model', metavar='M', type=str, 
                        default="MobileNet_V2", help='The name of the model to use')

    parser.add_argument('--pretrained', '-p', dest='pretrained', metavar='P', type=str, 
                        default='True', help='Using pretrained model')

    parser.add_argument('--wandb', '-w', dest='wandb', metavar='W', type=str, 
                        default='True', help='Using WandB')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    parser.add_argument('--checkpoint', '-c', dest='checkpoint', metavar='C', type=str, 
                        default='True', help='Saving checkpoints')

    parser.add_argument('--load', '-f', action='store_true', default=False, help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    # Test
    arg = get_args()
    print(arg.epochs)
    print(arg.wandb, type(arg.wandb))
    print(arg.amp, type(arg.amp))
    print(arg.load, type(arg.load))



