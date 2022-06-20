from pickletools import optimize
import torch




def loss_funcion(type='mse'):
    if type == 'mse':
        return torch.nn.MSELoss()
    if type == 'mae':
        return torch.nn.L1Loss()
    elif type == 'bce_wl':
        return torch.nn.BCEWithLogitsLoss()
    elif type == 'bce':
        return torch.nn.BCELoss()

def optimizer_loader(net, learning_rate: int = 0.001, amp: bool = False):

    # Defining the optimizer
    optimizer = torch.optim.Adam(
                            net.parameters(), 
                            lr=learning_rate, 
                            # betas=(0.5, 0.999), 
                            # weight_decay=1e-8,
                            )

    # Defining the scheduler
    # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    optimizer, 
                                                    'max', 
                                                    patience = 2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled = amp)

    return optimizer, scheduler, grad_scaler


