import torch
from torch.utils.data import Dataset
from PIL import Image
import os

from .common import generate_text_embeddings


# Map to turn binary attributes to something that resembles a sentence
template_map = {
    "5_o_Clock_Shadow": ("with a 5 o'clock shadow", ""),
    "Arched_Eyebrows": ("with arched eyebrows", ""),
    "Attractive": ("looking attractive", ""),
    "Bags_Under_Eyes": ("with bags under the eyes", ""),
    "Bald": ("who is bald", ""),
    "Bangs": ("with bangs", ""),
    "Big_Lips": ("with big lips", ""),
    "Big_Nose": ("with a big nose", ""),
    "Black_Hair": ("with black hair", ""),
    "Blond_Hair": ("with blond hair", ""),
    "Blurry": ("in a blurry photo", ""),
    "Brown_Hair": ("with brown hair", ""),
    "Bushy_Eyebrows": ("with bushy eyebrows", ""),
    "Chubby": ("who is chubby", ""),
    "Double_Chin": ("with a double chin", ""),
    "Eyeglasses": ("wearing eyeglasses", ""),
    "Goatee": ("with a goatee", ""),
    "Gray_Hair": ("with gray hair", ""),
    "Heavy_Makeup": ("wearing heavy makeup", ""),
    "High_Cheekbones": ("with high cheekbones", ""),
    "Male": ("man", "woman"),
    "Mouth_Slightly_Open": ("with mouth slightly open", ""),
    "Mustache": ("with a mustache", ""),
    "Narrow_Eyes": ("with narrow eyes", ""),
    "No_Beard": ("without a beard", ""),
    "Oval_Face": ("with an oval face", ""),
    "Pale_Skin": ("with pale skin", ""),
    "Pointy_Nose": ("with a pointy nose", ""),
    "Receding_Hairline": ("with a receding hairline", ""),
    "Rosy_Cheeks": ("with rosy cheeks", ""),
    "Sideburns": ("with sideburns", ""),
    "Smiling": ("smiling", ""),
    "Straight_Hair": ("with straight hair", ""),
    "Wavy_Hair": ("with wavy hair", ""),
    "Wearing_Earrings": ("wearing earrings", ""),
    "Wearing_Hat": ("wearing a hat", ""),
    "Wearing_Lipstick": ("wearing lipstick", ""),
    "Wearing_Necklace": ("wearing a necklace", ""),
    "Wearing_Necktie": ("wearing a necktie", ""),
    "Young": ("a young", "an old"),
}


def attributes_to_sentence(attr_vector, attributes):
    # Build sentence components
    active = [name for bit, name in zip(attr_vector, attributes) if bit == 1]

    gender = template_map['Male'][0] if 'Male' in active else template_map['Male'][1]
    age = template_map['Young'][0] if 'Young' in active else template_map['Young'][1]

    description_parts = [age, gender]

    for attr in active:
        if attr in template_map and attr not in ('Male', 'Young'):
            phrase = template_map[attr][0]
            if phrase:
                description_parts.append(phrase)

    return " ".join(description_parts)


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_df, attributes, clip_model, transform=None):
        self.img_dir = img_dir
        self.attr_df = attr_df
        self.transform = transform
        self.attributes = attributes
        self.clip_model = clip_model

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        filename = self.attr_df.iloc[idx].image_id
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        row = self.attr_df.iloc[idx]
        label = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)
        label = label + 1
        label = label // 2

        clip_embed = generate_text_embeddings(
                attributes_to_sentence(label, self.attributes),
                self.clip_model
                )

        return image, label, clip_embed
