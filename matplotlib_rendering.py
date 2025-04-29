import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from neural_image import *

from PIL import Image

plt.style.use('seaborn-v0_8-darkgrid')  # Sleek modern grid style

# Font & style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold'
})


class MatplotLibRendering:
    def __init__(self, model, losses=None):
        self.model = model

    def render(self, model_predictions, image_size, save_to_file=True):
        image = self.model.image
        size_x, size_y = image_size[0], image_size[1]
        predictions = model_predictions.reshape(size_y, size_x, 3)
        predictions = (predictions * 255).astype(np.uint8)

        if (save_to_file):
            im = Image.fromarray(predictions)
            im.save("result.png")

        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#f7f7f7')  # Soft background
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

        ax_pred = fig.add_subplot(gs[0, 0])
        self.render_prediction(ax_pred, predictions)

        ax_orig = fig.add_subplot(gs[0, 1])
        self.render_original(ax_orig, image)

        ax_loss = fig.add_subplot(gs[1, :])
        self.render_loss_graph(ax_loss)

        plt.tight_layout(pad=2)
        plt.show()
    
        

    def render_prediction(self, ax, predictions):
        ax.imshow(predictions, interpolation='bilinear')
        ax.set_title("Generated", fontsize=14, fontweight='semibold')
        ax.axis('off')
        self.add_frame(ax)

    def render_original(self, ax, image):
        ax.imshow(image)
        ax.set_title("Original", fontsize=14, fontweight='semibold')
        ax.axis('off')
        self.add_frame(ax)

    def render_loss_graph(self, ax):
        smoothed = self.smooth(self.model.get_losses(), window=10)
        ax.plot(smoothed, color='#ff4c4c', linewidth=2)
        ax.set_title("Loss Over Generations", fontweight='semibold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    

    def add_frame(self, ax):
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1)

    def smooth(self, data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')


if __name__ == "__main__":
    generator = NeuralImageGenerator(load_model=True)
    renderer = MatplotLibRendering(generator, generator.get_losses())
    predictions, size = generator.get_prediction(hyper_res=True)
    
    print(f"size: {size}")
    renderer.render(predictions, size, save_to_file=True)
