
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)




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
