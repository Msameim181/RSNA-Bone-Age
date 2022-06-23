
import logging
from pathlib import Path

import wandb

from utils.rich_logger import rich_print


def wandb_setup(config, notes: str = '') -> wandb:
    rich_print("\n[INFO]: Setting up WandB...")
    # Reading the config file, Key: Value
    f = open("temp/key/key.txt", "r")
    key = f.read()
    # Sign in to wandb
    wandb.login(key=key)

    model = config['model']
    run_name = config['name']
    device = config['device']
    dataset_name = config['dataset_name']
    # Create a run
    wandb.tensorboard.patch(root_logdir="./tensorboard")
    wandb_logger = wandb.init(
        project = "Bone-Age-RSNA", 
        entity = "rsna-bone-age", 
        sync_tensorboard = True,
        name = run_name, 
        tags = [
            'bone-age', 
            'rsna', 
            f'{model}', 
            f'{run_name}', 
            f'{device}',
            f'{dataset_name}'
        ],
        notes = notes)
    
    # wandb.tensorboard.patch(root_logdir="./tensorboard", tensorboard_x=False)
    # Configure wandb
    wandb_logger.config.update(config)
    # Logging
    rich_print("[INFO]: WandB setup completed.")
    return wandb_logger


def wandb_log_training_step(wandb_logger, loss, global_step, epoch, epoch_loss_step):
    # Logging
    wandb_logger.log({
        'Loss/Step Loss': loss,
        'Process/Step': global_step,
        'Loss/Train Loss (Step)': epoch_loss_step,
        'Process/Epoch': epoch
    })

def wandb_log_training(wandb_logger, epoch_loss, val_loss, epoch):
    # Logging
    wandb_logger.log({
        'Loss/Train Loss': epoch_loss,
        'Loss/Epoch Loss': epoch_loss,
        'Loss/Validation Loss (Epoch)': val_loss,
        'Process/Epoch': epoch,
    })

def wandb_log_histogram(net):
    # WandB Storing the model parameters
    histograms = {}
    for tag, value in net.named_parameters():
        tag = tag.replace('/', '.')
        histograms[f'Weights/{tag}'] = wandb.Histogram(value.data.cpu())
        histograms[f'Gradients/{tag}'] = wandb.Histogram(value.grad.data.cpu())

    return histograms

def wandb_log_validation(wandb_logger, optimizer, val_loss, acc, 
    images, batch_size, gender, boneage, age_pred, 
    global_step, epoch, histograms):
    # WandB Storing the results
    wandb_logger.log({
        'Process/Learning Rate': optimizer.param_groups[0]['lr'], 
        'Loss/Validation Loss': val_loss, 
        'Accuracy/Validation Correct': acc, 
        'Accuracy/Correct %': acc * 100,
        # Disable image uploading due to the size of the data and network traffic usage
        # 'Images': wandb.Image(images.cpu()) if batch_size == 1 else [wandb.Image(image.cpu()) for image in images], 
        'Gender': gender if batch_size == 1 else list(gender), 
        'Age': {
            'True': boneage.float().cpu().item() if batch_size == 1 else [age.float().cpu().item() for age in boneage], 
            'Pred': age_pred.float().cpu().item() if batch_size == 1 else [age.float().cpu().item() for age in age_pred],
        }, 
        'Process/Step': global_step, 
        'Process/Epoch': epoch, 
        # **histograms
        })


def wandb_log_evaluation(wandb_logger, result):
    wandb_logger.log({
            'Results/Evaluation First Loss': result['test_loss_first'],
            'Results/Evaluation Second Loss (MSE)': result['test_loss_second'],
            'Results/Evaluation Third Loss (MAE)': result['test_loss_third'],
            'Results/Evaluation Age Loss (MSE)': result['test_loss_mse_age'],
            'Results/Evaluation Age Loss (MAE)': result['test_loss_mae_age'],
            'Results/Evaluation Accuracy': result['accuracy'],
            'Results/Evaluation Correct': result['correct'],
    })
    wandb_logger.log({
            'Results/True Age All': result['boneage'], 
            'Results/Pred Age All': result['pred'], 
            'Results/Step All': list(range(len(result['pred'])))
    })
    for item, (t_age, p_age) in enumerate(zip(result['boneage'], result['pred'])):
        wandb_logger.log({
            'Results/True Age': t_age,
            'Results/Pred Age': p_age,
            'Results/Step': item
    })
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    # ax.plot(result['boneage'], 'r', label = 'True')
    # ax.plot(result['pred'], 'b', label = 'Pred')
    # wandb.log({"Results/Evaluaion Results": wandb.Image(fig)})


def wandb_log_model_artifact(wandb_logger, net_saved_path: str, run_name: str):
    model = wandb.Artifact(run_name, type='model')
    model.add_file(net_saved_path)
    wandb_logger.log_artifact(model)