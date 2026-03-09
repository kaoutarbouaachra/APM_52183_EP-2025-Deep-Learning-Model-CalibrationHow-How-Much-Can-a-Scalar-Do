# Evaluation of Calibration Methods

This folder implements the experiment described in **Section 4: The Role of Temperature Scaling**, whose goal is to compare the performance of several calibration methods on modern neural network architectures.

## Experimental Setup

We fine-tune **ResNet-152**, **MLP-Mixer**, and **ViT** on training sets of **45k images** from **CIFAR-10** and **CIFAR-100**. In addition, we evaluate the **zero-shot version of CLIP**, which does not require fine-tuning.

The model classes are located in the `models/` folder. The code used for fine-tuning is in `finetuning.py`, which relies on `train_model.py` to implement the training procedure. 

We use a **validation set of 5k images** and a **test set of 10k images**.

The notebook `summary.ipynb`, located outside this folder, contains the code for loading the checkpoints and evaluating the performance of the calibration methods.

First, we compute calibration metrics for the **uncalibrated models**. Then, we use the validation set to fit the calibration methods using **bins of equal length with $M = 10$**. The implemented methods (**Histogram Binning, Isotonic Regression, Matrix Scaling, Vector Scaling, and Temperature Scaling**) are available in `methods.py`.

Finally, we evaluate the calibrated models on the test set. The calibration metrics are implemented in `metrics.py`.

## Results 

| Dataset   | Model       | Uncalibrated | Hist. Binning | Isotonic Regr. | Matrix Scaling | Vector Scaling | Temp. Scaling |
|-----------|------------|--------------|---------------|----------------|---------------|---------------|---------------|
| CIFAR-10  | ResNet 152 | 3.42% | 0.79% | 0.86% | 0.81% | **0.44%** | 0.94% |
| CIFAR-10  | MLP-Mixer  | 3.85% | 1.3% | 1.18% | **0.7%** | 0.74% | 0.95% |
| CIFAR-10  | ViT        | 2.09% | 0.86% | 1.04% | 1.01% | **0.79%** | 0.84% |
| CIFAR-10  | CLIP       | 5.88% | 2.84% | 1.32% | 5.65% | **0.48%** | 2% |
| CIFAR-100 | ResNet 152 | 14.78% | 5.81% | 4.89% | 13.77% | **2.5%** | 4.32% |
| CIFAR-100 | MLP-Mixer  | 14.87% | 7.89% | 6.91% | 15.18% | **4.97%** | 5.21% |
| CIFAR-100 | ViT        | 10.64% | 6.08% | 4.62% | 11.88% | **3.31%** | 3.58% |
| CIFAR-100 | CLIP       | 10.47% | 7.6% | 3% | 15.09% | 2.58% | **2.38%** |

## References

1. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, *On Calibration of Modern Neural Networks*, Proceedings of the 34th International Conference on Machine Learning (ICML), pp. 1321–1330, 2017.

2. K. He, X. Zhang, S. Ren, and J. Sun, *Identity Mappings in Deep Residual Networks*, European Conference on Computer Vision (ECCV), 2016.

3. I. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, et al., *MLP-Mixer: An All-MLP Architecture for Vision*, Advances in Neural Information Processing Systems (NeurIPS), 2021.

4. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, International Conference on Learning Representations (ICLR), 2021.

5. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, et al., *Learning Transferable Visual Models From Natural Language Supervision*, OpenAI Technical Report, 2021.

6. M. P. Naeini, G. F. Cooper, and M. Hauskrecht, *Obtaining Well-Calibrated Probabilities Using Bayesian Binning*, AAAI Conference on Artificial Intelligence (AAAI), pp. 2901–2907, 2015.

7. B. Zadrozny and C. Elkan, *Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers*, Proceedings of the 18th International Conference on Machine Learning (ICML), pp. 609–616, 2001.

8. B. Zadrozny and C. Elkan, *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*, Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 694–699, 2002.