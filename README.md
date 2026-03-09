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
- **calibration_methods/**: Contains experiments illustrating the practical usefulness and effectiveness of Temperature Scaling.  
- **summary.ipynb**: A Jupyter notebook compiling all executed figures, results, and detailed analyses for easy review.  
---

## References

1. Guo, Chuan; Pleiss, Geoff; Sun, Yu; Weinberger, Kilian Q.  
   *On Calibration of Modern Neural Networks*, ICML 2017, pp. 1321–1330.  
   [Link](https://arxiv.org/abs/1706.04599)

2. Zhang, Hongyi; Cisse, Moustapha; Dauphin, Yann N.; Lopez-Paz, David.  
   *mixup: Beyond Empirical Risk Minimization*, arXiv 2017 / ICLR 2018.  
   [Link](https://openreview.net/forum?id=r1Ddp1-Rb)

3. Thulasidasan, Sunil; Bhattacharya, Chennupati; Bilmes, Jeff; Chen, Sungjin; Gupta, Jitendra.  
   *On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks*, NeurIPS 2019.  
   [Link](https://arxiv.org/abs/1905.11519)

4. Tolstikhin, Ilya; Houlsby, Neil; Kolesnikov, Alexander; Beyer, Lucas; et al.  
   *MLP-Mixer: An All-MLP Architecture for Vision*, NeurIPS 2021.  
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
   *Adam: A Method for Stochastic Optimization*, arXiv 2014.  
   [Link](https://arxiv.org/abs/1412.6980)

9. Zadrozny, Bianca; Elkan, Charles.  
   *Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers*, ICML 2001, pp. 609–616.

10. Zadrozny, Bianca; Elkan, Charles.  
    *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*, KDD 2002, pp. 694–699.

11. Naeini, Mahdi Pakdaman; Cooper, Gregory F.; Hauskrecht, Milos.  
    *Obtaining Well Calibrated Probabilities Using Bayesian Binning*, AAAI 2015, pp. 2901–2907.

12. Brier, Glenn W.  
    *Verification of Forecasts Expressed in Terms of Probability*, Monthly Weather Review 1950, 78(1):1–3.

13. Minderer, Matthias; Djolonga, Josip; Romijnders, Rob; Hubis, Frances; et al.  
    *Revisiting the Calibration of Modern Neural Networks*, NeurIPS 2021.

14. He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian.  
    *Identity Mappings in Deep Residual Networks*, ECCV 2016.

15. Zagoruyko, Sergey; Komodakis, Nikos.  
    *Wide Residual Networks*, BMVC 2016.

16. Huang, Gao; Liu, Zhuang; Weinberger, Kilian Q.; van der Maaten, Laurens.  
    *Densely Connected Convolutional Networks*, CVPR 2017.

17. Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander; et al.  
    *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.  
    [Link](https://openreview.net/forum?id=YicbFdNTTy)

18. Kolesnikov, Alexander; Beyer, Lucas; Zhai, Xiaohua; et al.  
    *Big Transfer (BiT): General Visual Representation Learning*, ECCV 2020.

19. Chen, Ting; Kornblith, Simon; Norouzi, Mohammad; Hinton, Geoffrey E.  
    *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020.

20. Radford, Alec; Kim, Jong Wook; Hallacy, Chris; et al.  
    *Learning Transferable Visual Models from Natural Language Supervision*, OpenAI 2021.  
    [Link](https://cdn.openai.com/papers/CLIP.pdf)

21. Deng, Jia; Dong, Wei; Socher, Richard; Li, Li-Jia; Li, Kai; Fei-Fei, Li.  
    *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR 2009, pp. 248–255.

22. Recht, Benjamin; Roelofs, Rebecca; Schmidt, Ludwig; Shankar, Vaishaal.  
    *Do ImageNet Classifiers Generalize to ImageNet?*, ICML 2019.

23. Krizhevsky, Alex; Hinton, Geoffrey.  
    *Learning Multiple Layers of Features from Tiny Images*, 2009.

24. Zhang, Jize; Kailkhura, Bhavya; Han, T. Yong-Jin.  
    *Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning*, ICML 2020, 119:11117–11128.  
    [Link](https://proceedings.mlr.press/v119/zhang20k.html)

25. Ming, Yifei; Fan, Ying; Li, Yixuan.  
    *How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?*, ICLR 2023.  
    [Link](https://openreview.net/forum?id=aEFaE0W5Hgt)

26. Ovadia, Yaniv; Fertig, Emily; Ren, Jie; et al.  
    *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS 2019.  
    [Link](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

27. Kull, Meelis; Perello-Nieto, Miquel; Kängsepp, Markus; et al.  
    *Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration*, NeurIPS 2019.  
    [Link](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9)



