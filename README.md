# Deep Learning Model Calibration: How Much Can a Scalar Do?

**Authors:** K. Bouaachra, A. Dahbi, A. Ghassoub, T. Meziane

## Abstract

While deep neural networks excel at generalizing to unseen data, they frequently exhibit a troubling tendency to assign high probability estimates to incorrect predictions. This phenomenon has spurred significant research into model calibration, aiming to align predicted confidences with actual accuracies.  

Among the proposed solutions, **Temperature Scaling** has emerged as a state-of-the-art post-hoc calibration method due to its simplicity and effectiveness in reducing overconfidence.  

In this work, we first demonstrate the practical utility of Temperature Scaling across various modern architectures. However, we show that this scalar transformation has fundamental boundaries. Specifically, we investigate two critical scenarios where its performance falters:  

1. **Class Overlap:** Where intrinsic regularization like Mixup proves more robust.  
2. **Distribution Shift:** Where the calibration parameters fail to generalize to out-of-distribution data.  

Our results highlight that while Temperature Scaling is a powerful baseline, it is not a universal remedy for uncertainty estimation in complex environments.

---
## Project Structure

The project is structured to separate experiments and analyses related to both the utility and limitations of Temperature Scaling:


- **Limitations/**: Contains experiments highlighting the boundaries of Temperature Scaling. Each subfolder includes a `README.md` explaining the experimental setup and methodology.  
- **Utility of Temperature Scaling/**: Contains experiments illustrating the practical usefulness and effectiveness of Temperature Scaling.  
- **final_notebook.ipynb**: A Jupyter notebook compiling all executed figures, results, and detailed analyses for easy review.  
---
## Usage

1. Clone the repository:  
```bash
git clone <repo-url>
cd APM_52183_EP-2025-Deep-Learning-Model-CalibrationHow-How-Much-Can-a-Scalar-Do

