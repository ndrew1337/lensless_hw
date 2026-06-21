# Lensless Imaging Homework

This is my solution for the homework on lensless image reconstruction — recovering images from a mask-based lensless camera ([DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)), where the lens is replaced by a coded mask and the image has to be deconvolved from the mask's PSF.

The original template for this project was made by Petr Grinberg here: [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) 

The task is described in more detail in [HW_README.md](HW_README.md).

## Methods

- **ADMM-100** — classical solver with fixed hyperparameters, no training
- **Unrolled ADMM-20** — ADMM with learnable per-iteration step sizes
- **Modular LeADMM-5** — a 5-iteration learnable ADMM wrapped by DRUNet pre- and/or post-processors

Bonuses:

- **FISTA** — a non-ADMM solver
- **Real-ESRGAN on ADMM-100** — a pretrained general-purpose GAN restorer run on top of vanilla ADMM-100, both frozen and fine-tuned on the training split

## Results

Per-image-mean metrics on the DigiCam test set (scale-aligned ROI). Speed is per 380×507 image at batch 1 on an RTX 5090.


| Method                                       | PSNR      | SSIM      | LPIPS     | MSE       | ms/img | params |
| -------------------------------------------- | --------- | --------- | --------- | --------- | ------ | ------ |
| ADMM-100                                     | 11.74     | 0.217     | 0.755     | 0.082     | 96     | —      |
| FISTA-100 *(bonus)*                          | 12.36     | 0.274     | 0.778     | 0.072     | 43     | —      |
| Unrolled ADMM-20                             | 11.67     | 0.193     | 0.755     | 0.082     | 34     | 80     |
| Modular: pre                                 | 13.27     | 0.144     | 0.660     | 0.054     | 26     | 8.2M   |
| Modular: post                                | 16.33     | 0.409     | 0.565     | 0.028     | 27     | 8.2M   |
| **Modular: pre + post**                      | **17.45** | **0.454** | **0.537** | **0.023** | 25     | 8.1M   |
| ADMM-100 + Real-ESRGAN *(bonus)*             | 11.23     | 0.241     | 0.744     | 0.092     | 295    | 16.7M  |
| ADMM-100 + Real-ESRGAN, fine-tuned *(bonus)* | 16.01     | 0.425     | 0.580     | 0.031     | 295    | 16.7M  |


## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train a model (`model` is one of `unrolled20`, `modular_pre`, `modular_post`, `modular_prepost`):

```bash
python train.py --config-name=lensless model=modular_prepost
```

Reconstruct and evaluate a trained model on the test set:

```bash
python inference.py --config-name=inference_admm100 model=modular_prepost \
    datasets=digicam_eval inferencer.skip_model_load=false \
    inferencer.from_pretrained=saved/<run_name>/model_best.pth
```

Classical solvers need no checkpoint — keep `inferencer.skip_model_load=true` and pass `model=admm100` or `model=fista`

Metrics from saved reconstructions, and the reconstruction-speed benchmark:

```bash
python calculate_metrics.py --gt <gt_dir> --pred <pred_dir>
python benchmark.py
```

Checkpoints are on HuggingFace: [PUFL/hw05-lensless](https://huggingface.co/PUFL/hw05-lensless)

## Notebooks

In the `notebooks/` folder:

- `demo.ipynb` — demo of the trained model 
- `kaggle_training.ipynb` — notebook for running training on Kaggle, made it for assistant's conveniency in checking that everything is working :)

## Report

This homework report: [BHW-2 Report on Weights & Biases](https://wandb.ai/andgri200-hse-university/hw05_lensless/reports/BHW-2-report--VmlldzoxNzI4OTMxNA?accessToken=8ukwfnjgvfx725w0pq25a4o0vr53oqga41gx4m442nal25wneo9859i1v1q63l2e)
