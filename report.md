# Text to image generation with CLIP

This project aims to implementa a text to image generation model using CLIP 
and a couple of image generation models. Our goal is to how different models
perform in this task.

# Architecture

// todo: this is just an outline, will improve it in the evening

## CLIP

A multimodal model developed by OpenAI, learns to understand images and text together​

It learns to give similar embeddings to matching image-text pairs​

We can use that to condition models during training

## Variational Auto-Encoder (VAE)

A VAE learns a latent distribution of training data. The Encoder learns to
encode an input image to a latent distribution.

Then the decoder tries to reconstruct that image by sampling random vector from $N(0, I)$.


## Conditional VAE (CVAE)

With VAE we can't control what image we get when we sample random vector. 
That's why we used Conditional VAE (CVAE)

We can use conditioning (e.g. concatenating a label or attribute vector to 
the input or latent vector) to generate data that corresponds to specific 
attributes.

## Generative Adversarial Network

# Data

## Dataset

We used CelebA dataset. It contains over 200000 portraits of celebrities, 
each with 40 binary attributes (male, wavy hair, smiling, wearing 
eyeglasses). We turned those attributes into prompts (with very simple mapping), the result was a simple text description for each image. It wasn't a perfect natural language sentence, but the method was pretty simple and straight forward so we decided to use it.

## Data preprocessing

To speed up training we resized images to 64x64 and normalized them. 

Since each image has a one-hot encoded attribute vector and we want a text
description of the image, we use give each image a description made up from
those attributes. This is done by turning each attribute into a part of a 
sentence ("Eyeglasses" -> "wearing eyeglasses", "High_cheekbones" -> "with high cheeckbones"). 
This was a straight forward method but it seemed enough for this usecase.

Also to save time during training all CLIP embeddings of those prompts were precomputed.


# Results

// todo: examples of generated images

# Conclusions 

// todo: what worked what didnt, challanges during training
