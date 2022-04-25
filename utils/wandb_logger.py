
import logging
import wandb






def wandb_setup(config) -> wandb:
    # Sign in to wandb
    wandb.login(key='0257777f14fecbf445207a8fdacdee681c72113a')

    
    model = config['model']
    run_name = config['name']
    device = config['device']
    # Create a run
    experiment = wandb.init(
        project = "Bone-Age-RSNA", 
        entity = "rsna-bone-age", 
        name = run_name, 
        tags = [
            'bone-age', 
            'rsna', 
            f'{model}', 
            f'{run_name}', 
            f'{device}'
        ],)
    # Configure wandb
    # experiment.config.update(dict(
    #     epochs = config['epochs'], 
    #     batch_size = config['batch_size'], 
    #     learning_rate = config['learning_rate'],
    #     save_checkpoint = config['save_checkpoint'], 
    #     amp = config['amp'],
    #     model = model,
    #     name = run_name,
    #     device = device))
    experiment.config.update(config)
    # Logging
    logging.info("WandB setup completed.")
    return experiment


def wandb_log_training_step(wandb_logger, loss, global_step, epoch, epoch_loss_step):
    # Logging
    wandb_logger.log({
        'Loss/Step Loss':            loss.item(),
        'Process/Step':                 global_step,
        'Loss/Train Loss (Step)':    epoch_loss_step,
        'Process/Epoch':                epoch
    })

def wandb_log_training(wandb_logger, epoch_loss, val_loss, epoch):
    # Logging
    wandb_logger.log({
        'Loss/Train Loss':               epoch_loss,
        'Loss/Epoch Loss':               epoch_loss,
        'Loss/Validation Loss (Epoch)':  val_loss,
        'Process/Epoch':                    epoch,
    })
    logging.info(f'\nEpoch: {epoch} | Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}\n')


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
        'Process/Learning Rate':        optimizer.param_groups[0]['lr'], 
        'Loss/Validation Loss':      val_loss, 
        'Accuracy/Validation Correct':   acc, 
        'Accuracy/Correct %':            acc * 100,
        'Images':               wandb.Image(images.cpu()) if batch_size == 1 else [wandb.Image(image.cpu()) for image in images], 
        'Gender':               gender if batch_size == 1 else list(gender), 
        'Age': {
            'True':             boneage.float().cpu() if batch_size == 1 else [age.float().cpu() for age in boneage], 
            'Pred':             age_pred.argmax(dim=1, keepdim=True)[0].float().cpu() if batch_size == 1 else [age for age in age_pred.argmax(dim=1, keepdim=True).float().cpu()],
        }, 
        'Process/Step':                 global_step, 
        'Process/Epoch':                epoch, 
        **histograms
    })