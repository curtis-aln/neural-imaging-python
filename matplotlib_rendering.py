import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class PredictionSlideshow:
    def __init__(self, predictions_list, sizes_list, training_images):
        self.predictions_list = predictions_list
        self.sizes_list = sizes_list
        self.training_images = training_images
        self.index = 0

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.subplots_adjust(bottom=0.2)

        # Buttons
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        self.update()

        plt.show()

    def update(self):
        pred = self.predictions_list[self.index].reshape(*self.sizes_list[self.index][::-1], 3)
        pred = (pred * 255).astype(np.uint8)
        orig = self.training_images[self.index]

        self.axs[0].imshow(pred)
        self.axs[0].set_title("Generated")
        self.axs[0].axis('off')

        self.axs[1].imshow(orig)
        self.axs[1].set_title("Original")
        self.axs[1].axis('off')

        self.fig.suptitle(f"Example {self.index + 1}/{len(self.predictions_list)}")
        self.fig.canvas.draw_idle()

    def next(self, event):
        self.index = (self.index + 1) % len(self.predictions_list)
        self.update()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.predictions_list)
        self.update()


