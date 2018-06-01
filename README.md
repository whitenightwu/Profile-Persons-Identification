# Profile-Persons-Identification

This is the implementation of the injection of the DREAM-block into the The Light CNN to improve the quality of recognition of extreme persons in the profile.

The CFP dataset is benchmark for recognition of extreme persons in the profile.

## Requirements
- NVIDIA GPU or CPU (GPU is prefered but only NVIDIA)
- Python 
- [Anaconda](https://docs.anaconda.com/anaconda/install/)
### Installation
- OpenCV
- CUDA (cuda 8.0 is preferred) CuDNN
- [PyTorch](https://pytorch.org/)

### Datasets
- Download face dataset such as  CASIA-WebFace, VGG-Face and [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/).
	- The MS-Celeb-1M clean list is uploaded: [Google Drive](https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing).

- [CFP dataset](http://www.cfpw.io/)

### Prepare

## Train

## Evaluate


## References
[Light CNN for Deep Face Recognition, in pytorch](https://github.com/AlfredXiangWu/LightCNN)
[The paper "A Light CNN for Deep Face Representation with Noisy Labels" is written by Xiang Wu, Ran He, Zhenan Sun and Tieniu Tan](https://arxiv.org/abs/1511.02683)
[DREAM block for Pose-Robust Face Recognition](https://github.com/penincillin/DREAM)
[The paper "Pose-Robust Face Recognition via Deep Residual Equivariant Mapping" is written by Yu Rong and Kaidi Cao](https://arxiv.org/abs/1803.00839)
[Frontal to Profile Face Verification in the Wild](http://www.cfpw.io/paper.pdf)