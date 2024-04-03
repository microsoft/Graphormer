# Distributional Graphormer: Towards Equilibrium Distribution Prediction for Molecular Systems with deep learning

This repository contains the code and data for the paper "Distributional Graphormer: Towards Equilibrium Distribution Prediction for Molecular Systems with deep learning", which is under review.

## Abstract

Advancements in deep learning have greatly improved the prediction of molecular system structures, such as accurately predicting protein structures from amino acid sequences. However, real-world applications extend beyond microscopic structure prediction and demand an understanding of how different structures are distributed at equilibrium. This comprehension is essential for bridging the gap between microscopic descriptors and macroscopic observations in accordance with statistical mechanics. Traditional methods for obtaining these distributions, such as molecular dynamics simulation, are often expensive and time-consuming. Addressing the challenges posed by the complex energy landscapes that govern probability distributions in high-dimensional space is crucial for predicting the equilibrium distribution across microscopic states. Inspired by the annealing process in thermodynamics, we introduce a novel deep learning framework, called Distributional Graphormer (DiG), to address these challenges. DiG employs deep neural networks to transform distributions from a simple form to the target equilibrium distribution, using microscopic descriptors of molecular systems as input. This framework enables efficiently generating diverse conformations and provides estimations of state densities. We demonstrate the performance of DiG on several molecular tasks, including protein conformation sampling, protein-ligand binding, molecular adsorption on catalyst surfaces, and property-guided structure sampling. DiG presents a significant advancement in statistically understanding microscopic molecules and predicting their macroscopic properties, opening up many exciting research opportunities in molecular science.

## Code and model

The code is organized into the following structure, with instructions on how to run the code in each subdirectory.

```
dig
├── catalyst-adsorption
├── property-guided
├── protein
└── protein-ligand
```

Each subdirectory contains datasets and checkpoints. Use the provided SAS token to download the files:

```bash
SAS="?sv=2021-10-04&st=2024-04-03T07%3A48%3A36Z&se=2025-04-04T07%3A48%3A00Z&sr=c&sp=rl&sig=TcaroWm0i9P2heueVCRbrUgTRX4SBbN%2BmpqlA3087JY%3D"
```

## Demo page

We have created a demo page for this paper, where you can explore some of the applications and visualizations of DiG. You can access the demo page at https://distributionalgraphormer.github.io/. The demo page showcases the results of DiG on protein conformation sampling, protein-ligand binding, and catalyst-adsorbate sampling. You can also interact with the models and generate new structures for different molecular systems. We hope you enjoy the demo and find it useful for your research.
