# Text to image generation with CLIP

Text to image model made in pytorch. This project uses CLIP model from OpenAI
with Conditional Variational Auto-Encoder to generate images from text prompts.
It was tested on CelebA dataset.

`run_training.sh` starts training in the background.

## Setup

1. Clone the repo and setup virtual environment

```bash
git clone https://github.com/your_username/clip-cvae.git
cd clip-cvae

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `my_secrets.py' (it is already in gitignore) file with following structure:

```python
import os

wandb_key = YOUR_KEY
wandb_proj_name = YOUR_PROJECT_NAME

data_path = PATH_TO_DATA
img_path = os.path.join(data_path, IMAGE_SUBDIRECTORY)
attr_path = os.path.join(data_path, ATTRIBUTE_SUBDIRECTORY)
```


- Use `run_training.sh` to start training

```
chmod +x run_training.sh
./run_training.sh
```
