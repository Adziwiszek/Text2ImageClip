import clip
import torch

device = 'cuda'

def generate_text_embeddings(text_prompts, clip_model):
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
    return text_embeddings.cpu()
