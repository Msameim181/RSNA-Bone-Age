import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the Your Model on images and target age.')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')

    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    parser.add_argument('--input-size', '-i', dest='input_size', metavar='I', type=int, default=2, help='Input size: gender or not')

    parser.add_argument('--num-classes', '-nc', dest='num_classes', metavar='NUM', type=int, default=0, help='Number of classes')

    parser.add_argument('--loss', '-loss', dest='loss_type', metavar='LOSS', type=str, default="bce_wl", help='Type of loss function (mse, bce_wl, bce, mae)')
    
    parser.add_argument('--out-type', '-ot', dest='output_type', metavar='OT', type=int, default=1, help='Output type (0:real, 1:onehot, 2:minmax, 3:zscore)')
    
    parser.add_argument('--gender-fc', '-gfc', dest='gender_fc_type', metavar='GFT', type=int, default=1, help='Output type (0: only 1 node append, 1: separate fc for gender append)')

    parser.add_argument('--attention', '-atten', action='store_true', default=False, help='Using attention models.')

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

    parser.add_argument('--dataset', '-d', dest='dataset', metavar='C', type=str, 
                        default='rsna', help='DataSet to use')

    parser.add_argument('--load', '-f', action='store_true', default=False, help='Load model from a .pth file')

    parser.add_argument('--bge', dest='basedOnSex', action='store_true', default=False, help='Based on gender.')

    parser.add_argument('--gender', '-g', dest='gender', type=str, 
                        default="male", help='The gender of dataset.')

    parser.add_argument('--note', '-n', dest='notes', type=str, 
                        default="", help='Run description for add to logger.')

    parser.add_argument('--name-suffix', '-ns', dest='name_suffix', type=str, 
                        default="", help='Suffix for the name of the run.')

    return parser.parse_args()


if __name__ == '__main__':
    # Test
    arg = get_args()
    print(type(arg))
    print(arg.epochs)
    print(arg.wandb, type(arg.wandb))
    print(arg.amp, type(arg.amp))
    print(arg.load, type(arg.load))
    print(arg.input_size, type(arg.input_size))
    print(arg.basedOnSex, type(arg.basedOnSex))
    print(arg.gender, type(arg.gender))
    print(arg.attention, type(arg.attention))
    # vars(arg)['kkp'] = "loop"
    # print(arg.kkp, type(arg.kkp))

    if arg.notes:
        print("hi")



