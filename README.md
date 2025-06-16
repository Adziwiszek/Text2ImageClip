# Text to image generation with CLIP

Text to image model made in pytorch. This project uses CLIP model from OpenAI
with Conditional Variational Auto-Encoder to generate images from text prompts.
It was tested on CelebA dataset.


## Structure

```bash
.
├── CVAE                    # Directory with CVAE model 
│   ├── __init__.py         
│   ├── cvae.py             # Main file with training loop 
│   ├── data_prep.py        # Dataset and stuff 
│   ├── model.py            # Model
│   └── common.py      
├── README.md
├── __init__.py
├── my_secrets.py           # File with keys and paths for this project 
└── run_cvae_training.sh    # Script for starting CVAE training
```


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
celeba_img_path = os.path.join(data_path, IMAGE_SUBDIRECTORY)
celeba_attr_path = os.path.join(data_path, ATTRIBUTE_SUBDIRECTORY)
```


3. Use `run_training.sh` to start training in the background

```bash
chmod +x run_training.sh
./run_training.sh
```

4. Or run directly:

```bash
python -m CVAE.cvae
```


