# VQGAN-from-scratch

> This is my attempt at coding VQGAN as presented in [Taming Transformers for High-Resolution Image Synthesis](https://compvis.github.io/taming-transformers/) from scratch.

VQGANs are improvement on VQVAE as presented in [Neural Discrete Representation Learning
](https://arxiv.org/abs/1711.00937) which use VAE to compresess data into a latent space but also add an additional constraint of limitting the latent representations to remain discrete in nature rather than continuous. This allows the model to capture meaningful features in a more structured manner.

> [!NOTE]
> This feels very similar to compressive sensing where we learn a dictionary to decompose signals as weighted atoms.