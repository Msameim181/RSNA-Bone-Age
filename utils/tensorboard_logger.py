import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.rich_logger import rich_print

# from tensorboardX import SummaryWriter


def tb_setup(config, args, log_dir:str = './tensorboard/', notes: str = '') -> SummaryWriter:
    """
    Setup tensorboard logger
    """
    if not log_dir:
        log_dir = './tensorboard'
    net = config['net']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    save_checkpoint = config['save_checkpoint']
    amp = config['amp']
    model = config['model']
    name = config['name']
    device = config['device']
    optimizer = config['optimizer']
    criterion = config['criterion']
    WandB_usage = config['WandB_usage']
    dataset_name = config['dataset_name']
    basedOnSex = config['basedOnSex']
    gender = config['gender']
    train_dataset_size = config['train_dataset_size']
    test_dataset_size = config['test_dataset_size']

    # Create a run
    tb_logger = SummaryWriter(log_dir=Path(log_dir, name))

    tb_logger.add_text(tag='notes', text_string=notes, global_step=0)
    tb_logger.add_text(tag='epochs', text_string=str(epochs), global_step=0)
    tb_logger.add_text(tag='batch_size', text_string=str(batch_size), global_step=0)
    tb_logger.add_text(tag='learning_rate', text_string=str(learning_rate), global_step=0)
    tb_logger.add_text(tag='save_checkpoint', text_string=str(save_checkpoint), global_step=0)
    tb_logger.add_text(tag='amp', text_string=str(amp), global_step=0)
    tb_logger.add_text(tag='model', text_string=str(model), global_step=0)
    tb_logger.add_text(tag='name', text_string=str(name), global_step=0)
    tb_logger.add_text(tag='device', text_string=str(device), global_step=0)
    tb_logger.add_text(tag='optimizer', text_string=str(optimizer), global_step=0)
    tb_logger.add_text(tag='criterion', text_string=str(criterion), global_step=0)
    tb_logger.add_text(tag='WandB_usage', text_string=str(WandB_usage), global_step=0)
    tb_logger.add_text(tag='dataset_name', text_string=str(dataset_name), global_step=0)
    tb_logger.add_text(tag='basedOnSex', text_string=str(basedOnSex), global_step=0)
    tb_logger.add_text(tag='gender', text_string=str(gender), global_step=0)
    tb_logger.add_text(tag='train_dataset_size', text_string=str(train_dataset_size), global_step=0)
    tb_logger.add_text(tag='test_dataset_size', text_string=str(test_dataset_size), global_step=0)

    if args.basedOnSex and args.input_size == 1:
        tb_logger.add_graph(net.cuda(), (torch.randn(batch_size, 1, 500, 625).cuda(), ))
    else:
        tb_logger.add_graph(net.cuda(), ([torch.randn(batch_size, 1, 500, 625).cuda(), torch.randn(batch_size, 1).cuda()], ))

    tb_logger.flush()

    return tb_logger



def tb_log_training_step(tb_logger, loss, global_step, epoch, epoch_loss_step):
     # Logging
    tb_logger.add_scalar('Loss/Step Loss', loss, global_step)
    tb_logger.add_scalar('Loss/Train Loss (Step)', epoch_loss_step, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)

    tb_logger.flush()


def tb_log_training(tb_logger, epoch_loss, val_loss, epoch):
    # Logging
    tb_logger.add_scalar('Loss/Train Loss', epoch_loss, epoch)
    tb_logger.add_scalar('Loss/Epoch Loss', epoch_loss, epoch)
    tb_logger.add_scalar('Loss/Validation Loss (Epoch)', val_loss, epoch)
    rich_print(f'\n[INFO]: Epoch: {epoch + 1} | Train Loss: {epoch_loss:.10f} | Validation Loss: {val_loss:.10f}\n')

    tb_logger.flush()
    

def tb_log_validation(tb_logger, optimizer, val_loss, acc, 
    images, batch_size, global_step, epoch, net):
    # TensorBoard Storing the results
    tb_logger.add_scalar('Process/Learning Rate', optimizer.param_groups[0]['lr'], global_step)
    tb_logger.add_scalar('Loss/Validation Loss (Step)', val_loss, global_step)
    tb_logger.add_scalar('Accuracy/Validation Correct (Step)', acc, global_step)
    tb_logger.add_scalar('Accuracy/Correct %', acc * 100, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)
    # img_batch = images.cpu() if batch_size == 1 else [image.cpu() for image in images]
    # tb_logger.add_images('Data/Images', img_batch, global_step)

    for name, param in net.named_parameters():
        tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

    tb_logger.flush()


def tb_log_evaluation(tb_logger, result):

    tb_logger.add_text(tag='Results/Evaluaion Loss (criterion)', text_string=str(result['test_loss_first']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluaion Loss Second (MSE)', text_string=str(result['test_loss_second']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluation Third Loss (MAE)', text_string=str(result['test_loss_third']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluation Age Loss (MSE)', text_string=str(result['test_loss_mse_age']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluation Age Loss (MAE)', text_string=str(result['test_loss_mae_age']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluaion Accuracy', text_string=str(result['accuracy']), global_step=0)
    tb_logger.add_text(tag='Results/Evaluaion Correct', text_string=str(result['correct']), global_step=0)

    for item, (t_age, p_age) in enumerate(zip(result['boneage'], result['pred'])):
        tb_logger.add_scalars('Results/Evaluaion Results', {
                'TrueAge': t_age,
                'PredAge': p_age,
            }, item)

    tb_logger.flush()

    
if __name__ == '__main__':
    # Test
    tb_setup(dict(
            net = MobileNet_V2(pretrained = True, image_channels = 1, num_classes = 229), 
            epochs = 1000, 
            batch_size = 2, 
            learning_rate = 0.001,
            save_checkpoint = './checkpoints/', 
            amp = False,
            model = "net.name",
            name = "run_namessss",
            device = "devicessss",
            optimizer = "optimizer.__class__.__name__",
            criterion = "criterion.__class__.__name__"))
