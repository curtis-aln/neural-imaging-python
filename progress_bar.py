import time
import sys

class ProgressBar:
    def __init__(self, total, prefix='', suffix='', length=40, fill='█', print_end='\r'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.start_time = time.time()
        self.last_print_length = 0

    def update(self, iteration):
        elapsed = time.time() - self.start_time
        progress = iteration / float(self.total)
        percent = f"{100 * progress:.1f}"
        filled_length = int(self.length * progress)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)

        if iteration > 0:
            est_total = elapsed / progress
            est_remaining = est_total - elapsed
        else:
            est_remaining = 0

        remaining_str = self.format_time(est_remaining)
        elapsed_str = self.format_time(elapsed)

        progress_string = f'{self.prefix} |{bar}| {percent}% {self.suffix} - {iteration}/{self.total} - ⏱ {elapsed_str} elapsed, ⌛ {remaining_str} left'

        # Clear previous line and overwrite
        sys.stdout.write('\r' + ' ' * self.last_print_length)
        sys.stdout.write('\r' + progress_string)
        sys.stdout.flush()
        self.last_print_length = len(progress_string)

        if iteration == self.total:
            print()  # Newline after final update

    @staticmethod
    def format_time(seconds):
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
