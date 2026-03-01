# Class Overlap Experiments

This folder contains experiments designed to investigate the **limitations of Temperature Scaling** in scenarios where classes overlap. The goal is to evaluate how well different models and calibration methods (ERM, Mixup, Temperature Scaling) handle **class overlap** situations, where the model might be more prone to overconfidence errors.

## Major Limitations of Temperature Scaling

### Class Overlap and the Advantage of Mixup

A key limitation of TS is its inability to correct overconfidence in regions where classes overlap. As demonstrated in prior studies, TS performance degrades proportionally to the degree of class overlap. In extreme cases, its effectiveness can approach that of random guessing, especially in high-dimensional feature spaces with many classes.

Under the ERM framework, the model is encouraged to assign maximum probability to the ground-truth label, often producing sharp decision boundaries even when classes are visually or semantically similar. TS, being a **global post-hoc adjustment**, applies the same temperature scalar to all predictions. This means it cannot distinguish between "clear" regions and "blurred" regions where classes intersect.

In contrast, **Mixup** acts during training as a form of vicinal risk minimization. By training the model to predict linear interpolations between classes, Mixup smooths the decision surface and encourages more conservative confidence estimates in overlapping regions. This intrinsic regularization addresses the root cause of overconfidence, achieving better calibration in scenarios where TS alone fails.
## Purpose

- Assess the robustness of Temperature Scaling when intrinsic class overlap is present.
- Compare standard training (ERM) and Mixup-augmented training.
- Study the effect of label noise on calibration performance.

## How to Run Experiments

Each experiment can be executed directly from the terminal using the `run_training.py` script. Below is a summary of the main commands:

| ID | Dataset | Model | Label Noise | Terminal Command | Done |
|----|---------|-------|------------|-----------------|------|
| 1  | CIFAR-10 | ResNet18 | 0.0 | `python3 run_training.py --task-name CIFAR10 --model-type ResNet18 --epochs 100 --label-noise 0.0 --save-model` 
| 2  | CIFAR-10 | ResNet18 | 0.1 | `python3 run_training.py --task-name CIFAR10 --model-type ResNet18 --epochs 100 --label-noise 0.1 --save-model` 
| 3  | CIFAR-10 | ResNet18 | 0.2 | `python3 run_training.py --task-name CIFAR10 --model-type ResNet18 --epochs 100 --label-noise 0.2 --save-model` 
| 4  | CIFAR-10 | MobileNetV2 | 0.0 | `python3 run_training.py --task-name CIFAR10 --model-type MobileNetV2 --epochs 100 --label-noise 0.0 --save-model` 
| 5  | CIFAR-10 | MobileNetV2 | 0.1 | `python3 run_training.py --task-name CIFAR10 --model-type MobileNetV2 --epochs 100 --label-noise 0.1 --save-model` 
| 6  | CIFAR-10 | MobileNetV2 | 0.2 | `python3 run_training.py --task-name CIFAR10 --model-type MobileNetV2 --epochs 100 --label-noise 0.2 --save-model` 
| 7  | CIFAR-10 | ResNext50 | 0.0 | `python3 run_training.py --task-name CIFAR10 --model-type ResNext50 --epochs 100 --label-noise 0.0 --save-model` 
| 8  | CIFAR-10 | ResNext50 | 0.1 | `python3 run_training.py --task-name CIFAR10 --model-type ResNext50 --epochs 100 --label-noise 0.1 --save-model` 
| 9  | CIFAR-10 | ResNext5à | 0.2 | `python3 run_training.py --task-name CIFAR10 --model-type ResNext5 --epochs 100 --label-noise 0.2 --save-model`
| 10 | CIFAR-100 | MobileNetV2 | 0.0 | `python3 run_training.py --task-name CIFAR100 --model-type MobileNetV2 --epochs 100 --label-noise 0.0 --save-model`
| 11 | CIFAR-100 | MobileNetV2 | 0.1 | `python3 run_training.py --task-name CIFAR100 --model-type MobileNetV2 --epochs 100 --label-noise 0.1 --save-model` 
| 12 | CIFAR-100 | MobileNetV2 | 0.2 | `python3 run_training.py --task-name CIFAR100 --model-type MobileNetV2 --epochs 100 --label-noise 0.2 --save-model` 
| 13  | CIFAR-10 | MobileNetV2 | 0.2 | `python3 run_training.py --task-name CIFAR10 --model-type MobileNetV2 --epochs 100 --label-noise 0.2 --save-model` 
| 14  | CIFAR-100 | ResNext50 | 0.0 | `python3 run_training.py --task-name CIFAR100 --model-type ResNext50 --epochs 100 --label-noise 0.0 --save-model` 
| 15  | CIFAR-100 | ResNext50 | 0.1 | `python3 run_training.py --task-name CIFAR100 --model-type ResNext50 --epochs 100 --label-noise 0.1 --save-model` 
| 16  | CIFAR-100 | ResNext5à | 0.2 | `python3 run_training.py --task-name CIFAR100 --model-type ResNext5 --epochs 100 --label-noise 0.2 --save-model` 
| 17  | CIFAR-100 | ResNet18 | 0.0 | `python3 run_training.py --task-name CIFAR100 --model-type ResNet18 --epochs 100 --label-noise 0.0 --save-model` 
| 18  | CIFAR-100 | ResNet18 | 0.1 | `python3 run_training.py --task-name CIFAR100 --model-type ResNet18 --epochs 100 --label-noise 0.1 --save-model` 
| 19  | CIFAR-100 | ResNet18 | 0.2 | `python3 run_training.py --task-name CIFAR100 --model-type ResNet18 --epochs 100 --label-noise 0.2 --save-model` 

## Experimental Configurations

| Feature | Values | Total |
|---------|--------|-------|
| Datasets | CIFAR-10, CIFAR-100 | 2 |
| Architectures | ResNet18, MobileNetV2, ResNeXt50 | 3 |
| Label Noise (η) | 0.0, 0.1, 0.2 | 3 |
| **Total Experiments** | **18 Configurations (ERM vs Mixup, with or without Temperature Scaling)** | **72 Models** |

Each experiment explores a different combination of dataset, architecture, and label noise, providing a systematic study of the **fragility of Temperature Scaling under class overlap conditions**.

## References

1. Guo, Chuan; Pleiss, Geoff; Sun, Yu; Weinberger, Kilian Q.  
   *On Calibration of Modern Neural Networks*, ICML 2017.  
   [Link](https://arxiv.org/abs/1706.04599)

2. Zhang, Hongyi; Cisse, Moustapha; Dauphin, Yann N.; Lopez-Paz, David.  
   *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.  
   [Link](https://openreview.net/forum?id=r1Ddp1-Rb)

3. Thulasidasan, Sunil; Bhattacharya, Chennupati; Bilmes, Jeff; Chen, Sungjin; Gupta, Jitendra.  
   *On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks*, NeurIPS 2019.  
   [Link](https://arxiv.org/abs/1905.11519)

4. Tolstikhin, Ilya; Houlsby, Neil; Kolesnikov, Alexander; Beyer, Lucas; et al.  
   *MLP-Mixer: An all-MLP Architecture for Vision*, NeurIPS 2021.  
   [Link](https://arxiv.org/abs/2105.01601)

5. Chidambaram, Muthu; Ge, Rong.  
   *On the Limitations of Temperature Scaling for Distributions with Overlaps*, ICLR 2024.  
   [Link](https://arxiv.org/abs/2306.00740)

6. Nixon, Jeremy; Dusenberry, Michael W.; Zhang, Linchao; Jerfel, Ghassen; Tran, Dustin.  
   *Measuring Calibration in Deep Learning*, CVPR Workshops 2019.  
   [Link](https://openaccess.thecvf.com/content_CVPRW_2019/html/CAL/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.html)

7. Küppers, Fabian; Schneider, Jan; Haselhoff, Anselm.  
   *Multivariate Confidence Calibration for Object Detection*, CVPR Workshops 2020.  
   [Link](https://openaccess.thecvf.com/content_CVPRW_2020/html/OD/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.html)

8. Kingma, Diederik P.; Ba, Jimmy.  
   *Adam: A Method for Stochastic Optimization*, 2014.  
   [Link](https://arxiv.org/abs/1412.6980)

