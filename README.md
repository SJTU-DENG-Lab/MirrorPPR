<p align="center">
  <img src="assets/mirrorppr_logo.png?raw=true" width="520" alt="MirrorPPR"/>
</p>

<h1 align="center">MirrorPPR: Exemplar-Based Portrait Photo Retouching</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.29308"><b>Paper</b></a> |
  <a href="https://sjtu-deng-lab.github.io/MirrorPPR"><b>Project Page</b></a> |
  <a href="https://github.com/SJTU-DENG-Lab/MirrorPPR"><b>Model</b></a> |
  <a href="https://github.com/SJTU-DENG-Lab/MirrorPPR"><b>Dataset</b></a>
</p>

## Abstract

While text-guided image editing has made remarkable progress, it remains limited in structural portrait retouching. Textual descriptions struggle to convey fine-grained changes to facial features and body proportions.
To address this gap, we introduce Exemplar-Based Portrait Photo Retouching, where the model is given an exemplar pair and tasked with inferring and applying the same retouching operations to a new query image.
Existing exemplar-based editing methods primarily focus on tasks with pronounced visual transformations. In contrast, structural portrait retouching involves extremely delicate and localized modifications, making accurate extraction and transfer of these edits challenging.
To tackle this, we propose MirrorPPR, a novel framework designed to capture and transfer subtle structural retouching operations. Our method uses a Retouching Operation Extractor to capture the subtle differences from the exemplar pair. The extracted representations are then injected into a pre-trained Diffusion Transformer (DiT) through a connector and Low-Rank Adaptation (LoRA) modules.
Furthermore, constructing perfectly aligned cross-identity training pairs is severely hindered by operation misalignment. To overcome this, we propose an advanced data self-augmentation paradigm that ensures strictly aligned retouching operations.
To alleviate data scarcity and support this novel task, we introduce MirrorPPR47M, a large-scale dataset with over 47 million retouched pairs. By structuring the dataset into simulated and professional subsets, we enable progressive curriculum learning to smoothly optimize the network.
Extensive experiments demonstrate that MirrorPPR significantly outperforms existing baselines in both retouching quality and identity preservation.

<p align="center">
  <img src="assets/model_architecture.png?raw=true" width="900" alt="MirrorPPR model architecture"/>
</p>


Code, models, and the dataset will be released soon.