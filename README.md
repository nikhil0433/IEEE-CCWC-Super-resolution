# CNN–ESPCNN Framework for Infrastructure Crack Detection and Super-Resolution

Research implementation accompanying **“Deep Learning Framework for Infrastructure Maintenance: Crack Detection and High-Resolution Imaging of Infrastructure Surfaces”** (IEEE CCWC 2027).

This repository contains a Google Colab notebook for a two-stage image-processing framework. A lightweight convolutional neural network (CNN) first classifies low-resolution concrete-surface images as crack-positive or crack-negative. An efficient sub-pixel convolutional neural network (ESPCNN) then performs **4× super-resolution only on images predicted to contain cracks**. The notebook also evaluates image quality and compares conditional super-resolution with processing every image.

## Framework

```text
Concrete surface image
         |
         v
   32 × 32 RGB input
         |
         v
 CNN crack classifier
         |
     p(crack) >= 0.5?
       /         \
     Yes          No
      |            |
      v            v
    ESPCNN      Bypass SR
    4× SR
      |
      v
 128 × 128 RGB output
```

The ESPCNN produces an enhanced RGB image; it does **not** perform crack segmentation or directly measure crack dimensions.

## Repository contents

| File | Description |
| --- | --- |
| [`CCWC2027_Full_CNN_ESPCNN_Colab(2).ipynb`](CCWC2027_Full_CNN_ESPCNN_Colab.ipynb) | End-to-end Google Colab experiment: dataset preparation, CNN and ESPCNN training, evaluation, plots, routing, MAC estimates, and inference-time benchmarks. |
| `README.md` | Project overview and instructions. |

The dataset, model checkpoints, and large result archives are **not required in the GitHub repository**; the notebook accesses the dataset and saves outputs using Google Drive.

## Dataset

The experiments use [Concrete Crack Images for Classification (Mendeley Data)](https://data.mendeley.com/datasets/5y9wdsg2zt/2), consisting of 40,000 concrete-surface patches (20,000 crack-positive and 20,000 crack-negative), each originally 227 × 227 RGB pixels. The dataset was generated from 458 source photographs.

The notebook samples **5,000 images per class**, using a fixed seed of `42`, then generates 32 × 32 low-resolution inputs. For the positive class, 128 × 128 reference images are also prepared for super-resolution evaluation.

| Experiment | Training | Validation | Testing |
| --- | ---: | ---: | ---: |
| CNN classification (both classes) | 4,900 | 2,100 | 3,000 |
| ESPCNN super-resolution (positive class) | 3,200 | 800 | 1,000 |
| Joint CNN → ESPCNN routing test | — | — | 2,000 |

The joint test comprises 1,000 crack-positive and 1,000 crack-negative patches held out from both models' training and validation partitions.

## Run in Google Colab

1. Download the notebook from this repository and upload it to [Google Colab](https://colab.research.google.com/).
2. Choose **Runtime → Change runtime type → GPU**. The notebook uses TensorFlow and installs additional Python packages in its first cell.
3. Download the dataset archive from Mendeley Data and place it in Google Drive at:

   ```text
   MyDrive/
   └── CCWC2027_CNN_ESPCNN_Results/
       └── 5y9wdsg2zt-2.zip
   ```

4. Run the notebook cells **in order**. Cell 1 mounts Google Drive; Cell 3 extracts the ZIP and any nested RAR archive and locates the `Positive` and `Negative` folders.
5. If your dataset is stored elsewhere, change `ZIP_PATH` in Cell 2 before running extraction.
6. The notebook writes checkpoints, CSV/JSON summaries, and plots to `MyDrive/CCWC2027_CNN_ESPCNN_Results/` and creates a results ZIP archive in `MyDrive/`.

> **Google Drive paths are configurable.** The notebook's `/content/...` paths are specific to the Colab runtime. Extracted images are stored on Colab's temporary disk, so they may need to be recreated after the runtime resets.

### Training configuration in the uploaded notebook

| Setting | CNN | ESPCNN |
| --- | --- | --- |
| Input | 32 × 32 RGB | 32 × 32 RGB |
| Output | Crack probability | 128 × 128 RGB |
| Trainable parameters | 6,177 | 83,376 |
| Loss | Binary cross-entropy | Mean squared error |
| Optimizer | Adam | Adam |
| Batch size | 32 | 16 |
| Maximum epochs | 2,000 | 2,000 |
| Early-stopping patience | 20 | 20 |
| Early-stopping `min_delta` | `0.0001` | `0.0001` |
| Learning rate | `1e-4`, then `1e-5` from epoch 100 | Same |

The learning-rate transition at epoch 100 is a documented implementation choice. The notebook currently uses `EARLY_STOP_MIN_DELTA = 0.0001`; change this configuration variable to investigate a different absolute validation-loss improvement threshold. `min_delta` is **not** a decimal-place rounding setting.

To train both models from scratch:

```python
RETRAIN_CNN = True
RETRAIN_ESPCNN = True
```

To reuse existing checkpoints in the configured results directory:

```python
RETRAIN_CNN = False
RETRAIN_ESPCNN = False
```

The expected checkpoint filenames are `cnn_best.weights.h5` and `espcnn_best.weights.h5`. Loading will fail if the files have not been saved in that directory. The notebook reconstructs each architecture before loading weights.

## Results from the recorded Colab run

### Crack classification (3,000 test images)

| Metric | Result |
| --- | ---: |
| Accuracy | 98.60% |
| Crack-positive precision | 99.32% |
| Crack-positive recall | 97.87% |
| Crack-positive F1 score | 98.59% |

Confusion-matrix counts: **1,468 true positives**, **1,490 true negatives**, **10 false positives**, and **32 false negatives**.

### Super-resolution (1,000 positive test images)

| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Bicubic interpolation | 28.153 | 0.7108 | 0.4115 |
| ESPCNN | **28.938** | **0.7241** | **0.3983** |

These values compare reconstructed 128 × 128 images with their reference images. Higher PSNR and SSIM, and lower LPIPS, indicate better image similarity.

### Selective processing (2,000-image joint test)

| Quantity | Result |
| --- | ---: |
| Images forwarded to ESPCNN | 983 |
| Images bypassing ESPCNN | 1,017 |
| Crack-positive images correctly forwarded | 976 |
| Crack-positive images missed | 24 |
| Crack-negative images incorrectly forwarded | 7 |
| Estimated forward-pass MAC reduction versus all-image ESPCNN | **46.15%** |
| Measured median GPU inference-time reduction | **4.08%** |

The GPU timing result reflects five trials under the notebook's benchmark settings and excludes image decoding, file I/O, and downstream detection. A reduction in the number of super-resolution operations **does not establish reduced downstream crack-detection false alarms**.

## Output files

The notebook saves results to the configured `RESULTS_DIR`. Key files include:

- `cnn_best.weights.h5`, `espcnn_best.weights.h5` — trained model checkpoints.
- `cnn_training_log.csv`, `espcnn_training_log.csv` — epoch-wise training histories.
- `Loss_Vs_Epochs.png`, `Accuracy_Vs_Epochs.png` — CNN training/validation curves.
- `ESPCNN_MSE_Vs_Epochs.png`, `ESPCNN_PSNR_Vs_Epochs.png` — ESPCNN training/validation curves.
- `cnn_test_predictions.csv`, `espcnn_test_per_image_metrics.csv` — test-level predictions and image-quality metrics.
- `joint_pipeline_routing.csv`, `model_MACs_per_layer.csv`, `inference_timing_trials.csv` — routing, computational complexity, and timing results.
- `FULL_EXPERIMENT_RESULTS.json` — consolidated experiment metrics and configuration information.

Other CSV files record the sampled images and the train/validation/test assignments.

## Scope and reproducibility notes

- **Patch-level separation:** Train, validation, and test sets contain distinct image patches, but their original source-photo identifiers were not recovered. Different patches from the same source photograph may therefore appear in separate partitions.
- **Synthetic low-resolution inputs:** The 32 × 32 images are generated by resizing existing images. Results may differ on native low-resolution photographs with camera blur, motion, or compression artifacts.
- **No downstream detector:** The study measures CNN screening, super-resolution quality, routing, and inference cost. It does not evaluate a post-super-resolution crack detector or segmentation network, so downstream false-alarm changes and crack geometry accuracy remain unverified.
- **Hardware-dependent timings:** GPU execution time and relative speedup depend on the runtime, batching, and device.
- **Training sensitivity:** In the recorded run, CNN trained for 168 epochs and ESPCNN for 63 epochs. The ESPCNN run ended before the scheduled learning-rate reduction at epoch 100; changing the early-stopping criterion could change the reported image-quality results.

## Dataset acknowledgment

Please cite the [original dataset and its associated publication](https://data.mendeley.com/datasets/5y9wdsg2zt/2) when using the concrete crack images. Dataset redistribution and use are governed by the dataset's own terms.

## Code availability

The notebook is provided for research reproducibility and inspection of the reported workflow. Check dataset licensing and repository contents before sharing files publicly; do not commit private Drive data, credentials, or large dataset archives.
