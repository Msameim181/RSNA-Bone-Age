
from datetime import datetime, timedelta
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

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


def train_progress_desc(speed, epoch_loss, step_loss):
    return f"[gold1]{speed:.2f}[white]img/s, Epoch Loss (Train)=[blue]{epoch_loss:.4f}[white], Step Loss (Batch)=[blue]{step_loss:.4f}[white]"

def valid_progress_desc(speed):
    return f"[gold1]{speed:.2f}[white]img/s"

def progress_get_speed(progress_bar, progress_id):
    return progress_bar._tasks[progress_id].speed or 0.00

def progress_get_data(progress_bar, progress_id):
    speed = progress_get_speed(progress_bar, progress_id)
    completed = progress_bar._tasks[progress_id].completed
    total = progress_bar._tasks[progress_id].total
    # finished_time = progress_bar._tasks[progress_id].finished_time
    elapsed = progress_bar._tasks[progress_id].elapsed
    fields = progress_bar._tasks[progress_id].fields
    # start_time = progress_bar._tasks[progress_id].start_time
    # stop_time = progress_bar._tasks[progress_id].stop_time
    percentage = progress_bar._tasks[progress_id].percentage
    # finished_speed = progress_bar._tasks[progress_id].finished_speed
    return completed, total, timedelta(seconds=int(elapsed)), fields, percentage, speed

def update_progress(bar_type, progress_bar, progress_id, advance, **kwargs):
    speed = progress_get_speed(progress_bar, progress_id)
    if bar_type:
        progress_bar.update(progress_id, 
            description = train_progress_desc(
                speed, 
                kwargs['epoch_loss'], 
                kwargs['step_loss']), 
            advance=advance)
    else:
        progress_bar.update(progress_id, 
            description=valid_progress_desc(speed), 
            advance=advance)

def result_progress(bar_type, progress_bar, progress_id, **kwargs):
    completed, total, elapsed, fields, percentage, speed = progress_get_data(progress_bar, progress_id)

    if bar_type:
        epoch_loss = kwargs['epoch_loss']
        step_loss = kwargs['step_loss']
        
        progress_bar.console.print(
            f"[green3]Training[white] Epoch [blue]{fields['epoch']+1}[white]/[blue]{fields['epochs']}[white]: "
            f"[not bold][orchid]{percentage:.0f}%[white][/not bold] [grey70]{completed}[white]/[grey70]{total}[white] "
            f"(Process Time: [cyan3]{elapsed}[white], Speed: [gold1]{speed:.2f}[white]img/s, "
            f"[not bold]Epoch Loss: [blue]{epoch_loss:.4f}[white], Last Step Loss (Batch): [blue]{step_loss:.4f}[white])[/not bold]"
        )
    
    else:
        val_loss = kwargs['val_loss']
        step_loss = kwargs['step_loss']
        val_repeat = kwargs['val_repeat']
        
        progress_bar.console.print(
            f"[bright_red]Validation[white] Epoch [blue]{fields['epoch']+1}[white] - Val [blue]{fields['val_round']+1}[white]/[blue]{val_repeat}[white]: "
            f"[not bold][orchid]{percentage:.0f}%[white][/not bold] [grey70]{completed}[white]/[grey70]{total}[white] "
            f"(Process Time: [cyan3]{elapsed}[white], Speed: [gold1]{speed:.2f}[white]img/s, "
            f"[not bold]Epoch Loss: [blue]{val_loss:.4f}[white], Last Step Loss (Batch): [blue]{step_loss:.4f}[white])[/not bold]"
        )

def stop_progress(progress_bar, progress_id, visible=False):
    progress_bar.stop_task(progress_id)
    progress_bar.update(progress_id, visible=visible)

def progress_group():
    return Group(
        train_progress,
        valid_progress
    )



def make_bar():
    return Progress(
        SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=50),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        "[progress.filesize]Passed: {task.completed} item",
        "•",
        "[progress.filesize.total]Total: {task.total} item",
    )

def make_console():
    return Console()