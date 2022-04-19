
import logging
import wandb



# sign in to wandb
# wandb.login(key='0257777f14fecbf445207a8fdacdee681c72113a')

def wandb_setup(config) -> wandb:
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