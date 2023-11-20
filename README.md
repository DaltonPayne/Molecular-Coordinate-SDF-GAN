# Molecular Coordinate SDF GAN

## Overview
This is an example approach to generating molecular structures using Generative Adversarial Networks (GANs). This project uses PyTorch for model architecture and RDKit for chemical informatics.

## Model
- **Training and Testing Data**: Molecular Descriptors from SDF files using RDKit.
- **Architecture**: PyTorch.

# Installation Instructions
To install the necessary packages, run the following commands in your terminal:

```bash
git clone https://github.com/[YourUsername]/MoleculeGAN.git
cd MoleculeGAN
pip install -r requirements.txt
```

Download a set of SDF files, either from PubChem or other sources for training and move them into the sdf folder. Run

```bash
python tensor.py
```
to convert the SDF files into tensors for training. Then use the jupyter notebook to train the model and generate molecular representation coordinates. You may also use train.py to train the model.