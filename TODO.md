# TODO List — neural-imaging-python

## 🧠 Core Features
- [ ] Make model class more configurable via arguments
- [ ] Add support for saving and loading models
- [ ] Implement model evaluation on custom test images

## 📈 Training & Logging
- [ ] Add training progress bar with `tqdm`
- [ ] Log training metrics to a file
- [ ] Enable training on GPU (check for CUDA)

## 🧪 Experimentation
- [ ] Try SIREN architecture with different activation functions
- [ ] Test performance on colored vs. grayscale images

## 📚 Documentation
- [ ] Write a detailed README with:
  - Project overview
  - Setup instructions
  - Example usage
- [ ] Add inline docstrings for all functions
- [ ] Create `requirements.txt` for Colab compatibility

## ✨ Nice-to-Haves
- [ ] Live loss plotting in Colab (e.g., with matplotlib or wandb)
- [ ] Support loading from a config file (e.g., `config.json`)
- [ ] Visual comparison: input vs. prediction side-by-side

## ✅ Completed
- [x] Set up initial model and training loop
- [x] Add basic image rendering from predictions
- [x] Move to Google Colab Pro for GPU support
