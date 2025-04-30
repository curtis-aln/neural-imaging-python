import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

plt.style.use('seaborn-v0_8-darkgrid')

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
        self.losses = losses or []

    def render(self, predictions_list, sizes_list, training_images, save_to_file=False, final_image_save_paths=None):
        num_images = len(predictions_list)
        final_image_save_paths = final_image_save_paths or [None] * num_images

        fig_height = max(6, num_images * 3)
        fig = plt.figure(figsize=(12, fig_height))
        fig.patch.set_facecolor('#f7f7f7')

        # 3 rows per image: [Prediction | Original] and one loss plot at the bottom
        outer_grid = gridspec.GridSpec(num_images + 1, 2, height_ratios=[3] * num_images + [1])

        zipped = zip(predictions_list, sizes_list, training_images, final_image_save_paths)
        for i, (preds, size, image, save_path) in enumerate(zipped):
            size_x, size_y = size
            prediction_img = preds.reshape(size_y, size_x, 3)
            prediction_img = (prediction_img * 255).astype(np.uint8)

            if save_to_file and save_path:
                Image.fromarray(prediction_img).save(save_path)

            ax_pred = fig.add_subplot(outer_grid[i, 0])
            self.render_prediction(ax_pred, prediction_img)

            ax_orig = fig.add_subplot(outer_grid[i, 1])
            self.render_original(ax_orig, image)

        ax_loss = fig.add_subplot(outer_grid[-1, :])
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
        if not self.losses:
            ax.set_visible(False)
            return
        smoothed = self.smooth(self.losses, window=10)
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
        return np.convolve(data, np.ones(window) / window, mode='valid')
