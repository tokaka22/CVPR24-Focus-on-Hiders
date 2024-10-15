
# HFAT-Pytorch Implementation

The official implementation of our CVPR 2024 paper "**Focus on Hiders: Exploring Hidden Threats for Enhancing Adversarial Training**" [[Arxiv](https://arxiv.org/html/2312.07067v1)] [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Focus_on_Hiders_Exploring_Hidden_Threats_for_Enhancing_Adversarial_Training_CVPR_2024_paper.html)]

## Usage

The experiments are conducted using with a single GeForce RTX 4090 24GB.
+ Create a virtual environment in terminal: `conda create -n HFAT python=3.8`.
+ Install necessary packages: `pip install -r requirements.txt`.
+ Download CIFAR-10 dataset and put it in `./datasets`.
+ Use the .sh file in `./scripts` to train.

## BibTeX

```
@InProceedings{Li_2024_CVPR, 
	author = {Li, Qian and Hu, Yuxiao and Dong, Yinpeng and Zhang, Dongxiao and Chen, Yuntian}, 
	title = {Focus on Hiders: Exploring Hidden Threats for Enhancing Adversarial Training}, 
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	month = {June}, 
	year = {2024}, 
	pages = {24442-24451}
}
```
```
@article{li2023focus,
  title={Focus on Hiders: Exploring Hidden Threats for Enhancing Adversarial Training},
  author={Li, Qian and Hu, Yuxiao and Dong, Yinpeng and Zhang, Dongxiao and Chen, Yuntian},
  journal={arXiv preprint arXiv:2312.07067},
  year={2023}
}
```
