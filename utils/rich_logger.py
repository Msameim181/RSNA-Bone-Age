# Libs
from datetime import timedelta

from rich.console import Group, Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

# The progress bar style for dataloader
data_progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn("[bold blue]{task.description}", justify="right"),
    BarColumn(bar_width=50, style='bar.back', complete_style='bar.complete', 
                finished_style='bar.finished', pulse_style='bar.pulse'),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
    "•",
    TimeElapsedColumn(),
    "•",
    "[progress.filesize]Passed: {task.completed} item",
    "•",
    "[progress.filesize.total]Total: {task.total:>.0f} item",
)

# The progress bar style for training
train_progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn(
        "[bold blue]Epoch {task.fields[epoch]}/{task.fields[epochs]}: [progress.percentage]{task.percentage:.0f}%"
    ),
    BarColumn(bar_width=50),
    TextColumn(
        "[green]{task.completed}/{task.total} [white]("
    ),
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    TextColumn(", {task.description})")
)

# The progress bar style for validation
valid_progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn(
        "[bold blue]Validation Round...: [progress.percentage]{task.percentage:.0f}%"
    ),
    BarColumn(bar_width=50),
    TextColumn(
        "[green]{task.completed}/{task.total} [white]("
    ),
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    TextColumn(", {task.description})")
)

# The progress bar style for evaluation
eval_progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn(
        "[bold blue]Evaluation Round...: [progress.percentage]{task.percentage:.0f}%"
    ),
    BarColumn(bar_width=50),
    TextColumn(
        "[green]{task.completed}/{task.total} [white]("
    ),
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    TextColumn(", {task.description})")
)

# Functions to create a rich logger
# Progress description is a string that will be displayed at the end of the progress bar
def train_progress_desc(speed: int, epoch_loss: float, step_loss: float, speed_unit: str = "img/s") -> str:
    """The progress Train description.

    Args:
        speed (int): The speed of the process.
        epoch_loss (float): The loss of the epoch. (sum / n)
        step_loss (float): The loss of the step.
        speed_unit (str, optional): The unit of the speed. Defaults to "img/s".

    Returns:
        str: The progress description.
    """
    return f"[gold1]{speed:.2f}[white]{speed_unit}, Epoch Loss (Train)=[blue]{epoch_loss:.4f}[white], Step Loss (Batch)=[blue]{step_loss:.4f}[white]"

def valid_progress_desc(speed: int, speed_unit: str = "img/s") -> str:
    """The progress Validation and Test description.

    Args:
        speed (int): The speed of the process.
        speed_unit (str, optional): The unit of the speed. Defaults to "img/s".

    Returns:
        str: The progress description.
    """
    return f"[gold1]{speed:.2f}[white]{speed_unit}"

def eval_progress_desc(speed: int, eval_loss: float, eval_acc: float, speed_unit: str = "img/s") -> str:
    """The progress Evaluation description.

    Args:
        speed (int): The speed of the process.
        eval_loss (float): The loss of the data. (sum / n)
        speed_unit (str, optional): The unit of the speed. Defaults to "img/s".

    Returns:
        str: The progress description.
    """
    return f"[gold1]{speed:.2f}[white]{speed_unit}, Data Loss (Test)=[blue]{eval_loss:.4f}[white], Accuracy=[gold1]{eval_acc:.2f}[white]%"

# Functions to process the update of the progress bar.
# Function to get the speed of the process
def progress_get_speed(progress_bar: Progress, progress_id: int) -> float:
    """The progress speed.
    
    Args:
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.
    
    Returns:
        float: The speed of the process.
    """
    return progress_bar._tasks[progress_id].speed or 0.00

def progress_get_data(progress_bar: Progress, progress_id: int) -> dict:
    """The progress data.

    Args:
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.

    Returns:
        dict: The progress data. (completed, total, description)
    """
    # Get the speed of the process
    progress_bar_data = {'speed': progress_get_speed(progress_bar, progress_id)}
    # Get the completed and total
    progress_bar_data['completed'] = progress_bar._tasks[progress_id].completed
    progress_bar_data['total'] = progress_bar._tasks[progress_id].total
    # Get the elapsed time
    progress_bar_data['elapsed'] = timedelta(seconds=int(progress_bar._tasks[progress_id].elapsed))
    # Get the input fields
    progress_bar_data['fields'] = progress_bar._tasks[progress_id].fields
    # Get the progress percentage
    progress_bar_data['percentage'] = progress_bar._tasks[progress_id].percentage
    # Get the description
    progress_bar_data['description'] = progress_bar._tasks[progress_id].description
    
    # Don't need the following fields
    # Get Start and Stop time
    progress_bar_data['start_time'] = progress_bar._tasks[progress_id].start_time
    progress_bar_data['stop_time'] = progress_bar._tasks[progress_id].stop_time
    # Get the finished time and speed
    progress_bar_data['finished_time'] = progress_bar._tasks[progress_id].finished_time
    progress_bar_data['finished_speed'] = progress_bar._tasks[progress_id].finished_speed

    return progress_bar_data

def update_progress_advance(progress_bar: Progress, progress_id: int, advance: int) -> int:
    """Calculate the advance of the progress bar based on the batch size, current step, and total steps.

    Args:
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.
        advance (int): The advance of the progress bar, for the next step.

    Returns:
        int: The advance of the progress bar.
    """
    next_meet = progress_bar._tasks[progress_id].completed + advance
    total = progress_bar._tasks[progress_id].total
    if next_meet > total:
        advance -= (next_meet - total)
    
    return advance

def update_progress(bar_type: int, progress_bar: Progress, progress_id: int, advance: int, **kwargs) -> None:
    """Update the progress bar.

    Args:
        bar_type (int): The type of the progress bar. (0: train, 1: valid, 2: evaluate (test))
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.
        advance (int): The advance of the progress bar, for the next step.
        **kwargs: The extra arguments of the progress bar. 
            For training progress bar, the extra arguments are:
                - step_loss: The loss of the step.
                - epoch_loss: The loss of the epoch.
    """
    # Get the progress speed
    speed = progress_get_speed(progress_bar, progress_id)
    # Calculate the advance of the progress bar
    advance = update_progress_advance(progress_bar, progress_id, advance)
    # Update the progress bar, based on the type of the progress bar.
    if bar_type == 0:
        progress_bar.update(progress_id, 
            description = train_progress_desc(speed, **kwargs), 
            advance = advance)

    elif bar_type == 1:
        progress_bar.update(progress_id, 
            description = valid_progress_desc(speed, **kwargs), 
            advance = advance)

    elif bar_type == 2:
        progress_bar.update(progress_id, 
            description = eval_progress_desc(speed, **kwargs), 
            advance = advance)
    
    return

def result_progress(bar_type: int, progress_bar: Progress, progress_id: int, **kwargs) -> None:
    """Show the result of the progress bar. After the progress bar is finished, the result will be shown.
    
    Args:
        bar_type (int): The type of the progress bar. (1: train, 0: valid and test)
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.
        **kwargs: The extra arguments of the progress bar.
            For training progress bar, the extra arguments are:
                - step_loss: The loss of the step.
                - epoch_loss: The loss of the epoch.

            For valid progress bar, the extra arguments are:
                - step_loss: The loss of the step.
                - val_loss: The loss of the validation.
                - val_acc: The accuracy of the validation.
                - val_repeat: The repeat of the validation.
            
    """
    progress_bar_data = progress_get_data(progress_bar, progress_id)
    # print the result, based on the type of the progress bar.
    if bar_type == 0:
        epoch_loss = kwargs['epoch_loss']
        step_loss = kwargs['step_loss']
        
        rich_print(progress_bar = progress_bar, 
            message = f"[green3]Training[white] Epoch [blue]{progress_bar_data['fields']['epoch']+1}[white]/[blue]{progress_bar_data['fields']['epochs']}[white]: "
            f"[not bold][orchid]{progress_bar_data['percentage']:.0f}%[white][/not bold] [grey70]{progress_bar_data['completed']}[white]/[grey70]{progress_bar_data['total']}[white] "
            f"(Process Time: [cyan3]{progress_bar_data['elapsed']}[white], Speed: [gold1]{progress_bar_data['speed']:.2f}[white]img/s, "
            f"[not bold]Epoch Loss: [blue]{epoch_loss:.4f}[white], Last Step Loss (Batch): [blue]{step_loss:.4f}[white])[/not bold]\n"
        )
    
    elif bar_type == 1:
        val_loss = kwargs['val_loss']
        step_loss = kwargs['step_loss']
        val_repeat = kwargs['val_repeat']
        val_acc = kwargs['val_acc']
        
        rich_print(progress_bar = progress_bar, 
            message = f"[orange1]Validation[white] Epoch [blue]{progress_bar_data['fields']['epoch']+1}[white] - Val [blue]{progress_bar_data['fields']['val_round']+1}[white]/[blue]{val_repeat}[white]: "
            f"[not bold][orchid]{progress_bar_data['percentage']:.0f}%[white][/not bold] [grey70]{progress_bar_data['completed']}[white]/[grey70]{progress_bar_data['total']}[white] "
            f"(Process Time: [cyan3]{progress_bar_data['elapsed']}[white], Speed: [gold1]{progress_bar_data['speed']:.2f}[white]img/s, "
            f"[not bold]Accuracy: [blue]{val_acc:.2f}[white]%, Epoch Loss: [blue]{val_loss:.4f}[white], Last Step Loss (Batch): [blue]{step_loss:.4f}[white])[/not bold]"
        )
    elif bar_type == 2:
        eval_loss = kwargs['eval_loss']
        step_loss = kwargs['step_loss']
        eval_acc = kwargs['eval_acc']
        
        rich_print(progress_bar = progress_bar, 
            message = f"\n[bright_red]Evaluation[white]: [not bold][orchid]{progress_bar_data['percentage']:.0f}%[white][/not bold] "
            f"[grey70]{progress_bar_data['completed']}[white]/[grey70]{progress_bar_data['total']}[white] (Process Time: [cyan3]{progress_bar_data['elapsed']}[white], "
            f"Speed: [gold1]{progress_bar_data['speed']:.2f}[white]img/s, [not bold]Accuracy: [blue]{eval_acc:.2f}[white]%, Data Loss: [blue]{eval_loss:.4f}[white], "
            f"Last Step Loss (Batch): [blue]{step_loss:.4f}[white])[/not bold]"
        )
    return

def stop_progress(progress_bar: Progress, progress_id: int, visible: bool = False) -> None:
    """Stop the progress bar.
    
    Args:
        progress_bar (Progress): The progress bar.
        progress_id (int): The progress id.
        visible (bool): Whether the progress bar is visible after ending or not.
        
    """
    progress_bar.stop_task(progress_id)
    progress_bar.update(progress_id, visible=visible)
    return


def progress_group() -> Group:
    """Create a group of progress bars. for initializing the progress bars.

    Returns:
        Group: Group of the progress bars: train and valid bars.
    """
    return Group(
        train_progress,
        valid_progress,
        eval_progress
    )

def rich_print(message: str, progress_bar: Progress = train_progress, console: Console = Console()) -> None:
    """Print the message with rich style. Give me anything with rich style."""
    if progress_bar.tasks:
        progress_bar.console.print(f"{message}")
    else:
        console.print(f"{message}")
    return


def make_console():
    """Simple console for printing."""
    return Console()



if __name__=='__main__':
    rich_print("[red]Hello World![white]")
    train_progress.add_task(description="Starting...", total=5, epoch = 1, epochs = 10)
    rich_print("[red]Hello World![white]")