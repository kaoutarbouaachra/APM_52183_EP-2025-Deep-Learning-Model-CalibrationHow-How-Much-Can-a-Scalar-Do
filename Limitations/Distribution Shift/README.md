# Post-Hoc Calibration Under Distribution Shift (Ovadia+ Benchmark)

This repository contains experiments designed to investigate the **limitations of Temperature Scaling (TS)** and evaluate modern post-hoc calibration methods under **distribution shift**. The goal is to understand how different calibrators (TS, ETS, TvA, DAC) behave when a model is exposed to corrupted, out-of-distribution (OOD) data where it is highly prone to overconfidence errors.

## Major Limitations of Temperature Scaling under Shift

### The Ovadia Collapse and the Advantage of ETS

A key limitation of Temperature Scaling (TS) is its reliance on a single, global scalar fitted on clean validation data. As demonstrated by Ovadia et al. (2019), TS performance degrades catastrophically under domain shift (e.g., image corruptions). Because TS has no mechanism to detect OOD inputs, corrupted images producing noisy logits receive the exact same temperature scaling as clean images, resulting in highly confident but incorrect predictions (severe NLL degradation).

In contrast, **Ensemble Temperature Scaling (ETS)** acts as a structural safeguard. By hedging the temperature-scaled softmax with the original distribution and a uniform prior, ETS explicitly bounds the maximum confidence the model can output under extreme uncertainty. This addresses the root cause of TS failure under shift, achieving vastly superior Negative Log-Likelihood (NLL) without requiring access to corrupted data at calibration time.

### The TvA Paradox: Low ECE $\neq$ Good Calibration

Our experiments expose a critical flaw in relying solely on Expected Calibration Error (ECE). **Top-versus-All (TvA)** calibration achieves near-zero ECE under severe noise corruptions, creating the illusion of perfect calibration. However, analysis of NLL reveals that TvA's per-class sigmoid normalisation aggressively squashes all predictions toward uniform probability ($1/K$). The low ECE simply arises from the model's confidence collapsing at the exact same rate as its accuracy, effectively masking a complete loss of discriminative power.

## Purpose

- Assess the robustness of standard Temperature Scaling against data distribution shifts.
- Compare standard TS with state-of-the-art post-hoc alternatives: ETS, TvA, and Density-Aware Calibration (DAC).
- Systematically evaluate these methods across 13 corruption types and 5 severity levels (CIFAR-100-C taxonomy).
- Demonstrate the necessity of evaluating both ECE and NLL to detect pathological calibration artifacts.


## Experimental Configurations

| Feature | Details | Total |
| :-- | :-- | :-- |
| Dataset | Clean CIFAR-100 + CIFAR-100-C | 1 + 13 |
| Architecture | DenseNet-BC-40 (Trained from scratch via ERM) | 1 |
| Corruptions | Noise (Gaussian, Shot, Impulse), Blur (Defocus, Glass, Motion, Zoom), Weather (Fog, Brightness, Contrast), Digital (Elastic, Pixelate, JPEG) | 13 |
| Severities | Levels 1 through 5 | 5 |
| Calibrators | TS, ETS, TvA, DAC | 4 |
| **Total Evaluation Runs** | **4 methods $\times$ 13 corruptions $\times$ 5 severities** | **260 evaluations** |

## How to Run Experiments

The experimental pipeline is split into model training, calibrator fitting, and distribution shift evaluation.

### 1. Train the Base Model

First, train a standard DenseNet-BC-40 on CIFAR-100. This will output `model.pth` and `valid_indices.pth` to the `.checkpoints/` directory.


| Action | Terminal Command |
| :-- | :-- |
| Train DenseNet-40 | `python3 train.py --data .data --save .checkpoints --depth 40 --nepochs 300` |

### 2. Run the Distribution Shift Benchmark

Once the model is trained, evaluate all four post-hoc calibrators across the entire corruption taxonomy. The script will automatically load the validation set used during training to fit the calibrators, apply the 13 corruptions on-the-fly, and save the metrics.


| Action | Terminal Command |
| :-- | :-- |
| Run Full Benchmark | `python3 evaluate_shift.py --data .data --save .checkpoints` |

*Note: Running this command will generate `.checkpoints/ovadia_plus_results.json`, which contains the complete breakdown of Accuracy, NLL, ECE, and Adaptive ECE for every method, corruption, and severity level.*

### 3. Analyze and Visualize Results

Open the provided Jupyter Notebook to visualize the benchmark results.


| Action | Command / File |
| :-- | :-- |
| Generate Reliability Diagrams | Run all cells in `summary.ipynb` |
| View Individual Corruptions | `python3 plot_per_corruption.py` |

## Repository Structure

- `train.py`: Script to train the base DenseNet-BC-40 model.
- `models/`: Contains the DenseNet architecture definitions.
- `corruptions.py`: On-the-fly generation of CIFAR-100-C image corruptions.
- `temperature_scaling.py`: Implementations of `ModelWithTemperature` (TS), `ETS`, `TvA`, and `DAC`.
- `evaluate_shift.py`: Main benchmarking loop applying the corruptions and recording metrics to JSON.
- `plot_per_corruption.py`: Generates line plots of calibration metrics vs corruption severity.
- `summary.ipynb`: Comprehensive Jupyter notebook analyzing the JSON results, plotting metric trajectories, and generating fixed-grid reliability diagrams.


## References

1. Guo, C., Pleiss, G., Sun, Y., \& Weinberger, K. Q. (2017).
*On Calibration of Modern Neural Networks*. ICML.
[Link](https://arxiv.org/abs/1706.04599)
2. Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., ... \& Snoek, J. (2019).
*Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*. NeurIPS.
[Link](https://arxiv.org/abs/1906.02530)
3. Hendrycks, D., \& Dietterich, T. (2019).
*Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR.
[Link](https://arxiv.org/abs/1903.12261)
4. Zhang, J., Kailkhura, A., \& Han, T. Y. J. (2020).
*Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning*. ICML.
*(Introduces ETS)* [Link](https://arxiv.org/abs/2003.07329)
5. Paisley, J., et al. (2024).
*Top-versus-All Calibration for Neural Networks*.
6. *Density-Aware Calibration (DAC)* (2023).
*(Input-dependent scaling using logit-space L2 distance to validation mean).*
<span style="display:none">[^1][^2][^3][^4][^5][^6][^7]</span>

<div align="center">⁂</div>

[^1]: corruptions.py

[^2]: demo.py

[^3]: train.py

[^4]: temperature_scaling.py

[^5]: evaluate_shift.py

[^6]: plot_per_corruption.py

[^7]: summary.ipynb
