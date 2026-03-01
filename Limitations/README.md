# Limitations of Temperature Scaling

This folder contains experiments designed to explore the **limitations** of Temperature Scaling in model calibration. While Temperature Scaling is effective in reducing overconfidence, it has certain boundaries that this project investigates.

## Folder Overview

- **Class Overlap/**  
  Contains experiments where classes have significant overlap. Mixup or intrinsic regularization methods are shown to be more robust in these scenarios. See the `README.md` inside for detailed methodology and results.

- **Distribution Shift/**  
  Contains experiments testing Temperature Scaling under distribution shifts. Calibration parameters often fail to generalize to out-of-distribution data. See the `README.md` inside for detailed methodology and results.

## How to Use

- Explore each subfolder for the detailed experiment descriptions.
- Each subfolder's `README.md` explains the experimental setup and methodology.
