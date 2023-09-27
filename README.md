Code for "Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion" coming soon!

# Contrastive-Lift (NeurIPS 2023 Spotlight)
## [[Project page](https://www.robots.ox.ac.uk/~vgg/research/contrastive-lift/)] [[Paper](https://arxiv.org/abs/2306.04633)]

> **Yash Bhalgat**, Iro Laina, João F. Henriques, Andrew Zisserman, Andrea Vedaldi
> 
> Instance segmentation in 3D is a challenging task due to the lack of large-scale annotated datasets. In this paper, we show that this task can be addressed effectively by leveraging instead 2D pre-trained models for instance segmentation. We propose a novel approach to lift 2D segments to 3D and fuse them by means of a neural field representation, which encourages multi-view consistency across frames. The core of our approach is a slow-fast clustering objective function, which is scalable and well-suited for scenes with a large number of objects. Unlike previous approaches, our method does not require an upper bound on the number of objects or object tracking across frames. To demonstrate the scalability of the slow-fast clustering, we create a new semi-realistic dataset called the Messy Rooms dataset, which features scenes with up to 500 objects per scene. Our approach outperforms the state-of-the-art on challenging scenes from the ScanNet, Hypersim, and Replica datasets, as well as on our newly created Messy Rooms dataset, demonstrating the effectiveness and scalability of our slow-fast clustering method.

![image](https://github.com/yashbhalgat/Contrastive-Lift/assets/8559512/30f48101-548d-4d79-857f-b64b64087e66)


# Citation
If you find this work useful in your research, please cite our paper:
```
@inproceedings{
  bhalgat2023contrastive,
  title={Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion},
  author={Bhalgat, Yash and Laina, Iro and Henriques, Jo{\~a}o F and Zisserman, Andrew and Vedaldi, Andrea},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=bbbbbov4Xu}
}
```
