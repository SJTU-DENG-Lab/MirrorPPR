<p align="center">
  <img src="assets/mirrorppr_logo.png?raw=true" width="520" alt="MirrorPPR"/>
</p>

<h1 align="center">MirrorPPR: Exemplar-Based Portrait Photo Retouching</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.29308"><b>Paper</b></a> |
  <a href="https://sjtu-deng-lab.github.io/MirrorPPR"><b>Project Page</b></a> |
  <a href="https://huggingface.co/SJTU-DENG-Lab/MirrorPPR-Face"><b>Model</b></a> |
  <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/MirrorPPR47M"><b>Dataset</b></a> |
  <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/SimFace-100"><b>Benchmark</b></a>
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

## Installation

```bash
conda env create -f environment.yml
conda activate mirrorppr
```

## MirrorPPR-Face on SimFace-100

Download the released MirrorPPR-Face weights and SimFace-100 benchmark:

```bash
huggingface-cli download SJTU-DENG-Lab/MirrorPPR-Face \
  --local-dir checkpoints/MirrorPPR-Face

huggingface-cli download SJTU-DENG-Lab/SimFace-100 \
  --repo-type dataset \
  --local-dir data/SimFace-100
```

Download the InsightFace `buffalo_l` model pack for identity evaluation:

```bash
mkdir -p checkpoints/insightface/models
wget -O checkpoints/insightface/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip checkpoints/insightface/buffalo_l.zip -d checkpoints/insightface/models
```

Run inference:

```bash
python mirrorppr/inference/simface100.py \
  --dataset-root data/SimFace-100 \
  --weights-root checkpoints/MirrorPPR-Face \
  --output-dir outputs/simface100_mirrorppr_face \
  --gpus 0,1,2,3,4,5,6,7 \
  --steps 40
```

Run evaluation:

```bash
python mirrorppr/eval/evaluate.py \
  --log-file outputs/simface100_mirrorppr_face/inference_log.json \
  --output-file outputs/simface100_mirrorppr_face/evaluation_metrics.json \
  --insightface-root checkpoints/insightface
```

## MirrorPPR47M Construction

Download the released MirrorPPR47M metadata and assets:

```bash
huggingface-cli download SJTU-DENG-Lab/MirrorPPR47M \
  --repo-type dataset \
  --local-dir data/MirrorPPR47M
```

### Simulated Subset

Download FFHQ aligned images and in-the-wild images following https://github.com/NVlabs/ffhq-dataset.
The simulated construction script applies LLW-style operations to one aligned FFHQ image, pastes the retouched face back into the wild image, and generates paired wild-frame augmentations.

```bash
python mirrorppr/data/simulated_subset.py \
  --aligned-image /path/to/ffhq/images1024x1024/34714.png \
  --wild-image /path/to/ffhq/in-the-wild-images/34714.png \
  --ffhq-metadata data/MirrorPPR47M/simulated/ffhq-dataset-v2.json \
  --image-id 34714 \
  --operations eye_resize:100,nose_alar:-100 \
  --output-dir outputs/simulated_sample
```

### Professional Subset

For a PPR10K image, first list the executable atomic operations in the released professional subset. The script prints the list and also writes `<image_id>_professional_operations.json` by default.

```bash
python mirrorppr/data/list_professional_operations.py \
  --single-op-json data/MirrorPPR47M/professional/metadata/single_ops_all.json \
  --data-root data/MirrorPPR47M \
  --image-id 1443:3
```

The listed `operation_id` values are single operations. To construct a professional sample, freely combine any compatible operations for the same image by passing comma-separated ids to `--operations`. The construction script composes the selected operation tiles into a full-resolution retouched image, then generates the crop and augmentations.

```bash
python mirrorppr/data/professional_subset.py \
  --single-op-json data/MirrorPPR47M/professional/metadata/single_ops_all.json \
  --data-root data/MirrorPPR47M \
  --image-id 1443:3 \
  --operations high_mouth:100,body_shape_thin_shoulders:-100 \
  --output-dir outputs/professional_sample
```

## Acknowledgements

We would like to sincerely thank the developers of [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Moto](https://github.com/TencentARC/Moto), [PPR10K](https://github.com/csjliang/PPR10K), and [FFHQ](https://github.com/NVlabs/ffhq-dataset), as our work is heavily built upon these resources.

## Citation

```bibtex
@misc{liu2026mirrorpprexemplarbasedportraitphoto,
      title={MirrorPPR: Exemplar-Based Portrait Photo Retouching}, 
      author={Zhihong Liu and Zheng Li and Jiachun Jin and Siqi Kou and Yitao Jian and Fengpei Yu and Zhijie Deng},
      year={2026},
      eprint={2606.29308},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.29308}, 
}
```
