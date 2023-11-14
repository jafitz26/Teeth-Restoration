# HDTR: A Real-Time High-Definition Teeth Restoration Network for Arbitrary Talking Face Generation Methods

We propose `A Real-Time High-Definition Teeth Restoration Network (HDTR-Net)` to address talking face videos with blurred mouth 
in this work, which aims to improve clarity of talking face mouth and lip regions in real-time inference 
that correspond to given arbitrary talking face videos.

[[Paper]](https://arxiv.org/abs/2309.07495)

<img src='./docs/img/HDTR-Net.png' width=880>

### Arguments

Here's a breakdown of what each argument controls:

--mask_channel: This argument specifies the channel of the mouth mask and mouth contour. The default value is 6. In the context of image processing, a channel refers to the grayscale image of the same size as a color image, made up of just one of these primary colors. For instance, an image from a standard digital camera will have three channels: red, green, and blue.

--ref_channel: This argument specifies the reference channel. The default value is 3. The exact meaning of "reference channel" would depend on the context of your project, but it generally refers to a specific channel in an image or data set that's used as a reference or comparison point for other data.

--mouth_size: This argument specifies the size of the cropped mouth region image. The default value is 96. This likely refers to the size (in pixels) of the square region of interest around the mouth in an image.

--test_batch_size: This argument specifies the size of the test batch. The default value is 8. In the context of machine learning, a batch is a subset of the dataset used for training or testing. The batch size is the number of samples processed before the model is updated.

--test_workers: This argument specifies the number of workers to run the test. The default value is 0. In the context of PyTorch (a popular machine learning library in Python), a worker refers to a subprocess that loads data from disk and puts it on the CPU memory. The number of workers determines how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

### Recommondation of our works
This repo is maintaining by authors, if you have any questions, please contact us at issue tracker.

**The official repository with Pytorch**
**Our method can restorate teeth region for arbitrary face generation on images and videos**

## Test Results
![Results1](./docs/img/male.png)
![Results2](./docs/img/male_HQ.png)
![Results3](./docs/img/female_side-face.png)
![Results4](./docs/img/female_HQ_side-face.png)

## Requirements
* [python](https://www.python.org/download/releases/)（We use version 3.7)
* [PyTorch](https://pytorch.org/)（We use version 1.13.1)
* [opencv2](https://opencv.org/releases.html)
* [ffmpeg](https://ffmpeg.org/)

We conduct the experiments with 4 32G V100 on CUDA 10.2. For more details, please refer to the `requirements.txt`. We recommend to install [pytorch](https://pytorch.org/) firstly, and then run:
```
pip install -r requirements.txt
```

## Generating test results
* Download the pre-trained model [checkpoint](https://drive.google.com/drive/folders/1IGJpQVC2fbJJASoS7bbPdt722vSvMtHr?hl=zh-cn) 
Create the default folder `./checkpoint` and put the checkpoint in it or get the CHECKPOINT_PATH, Then run the following 

bash
``` 
CUDA_VISIBLE_DEVICES=0 python inference.py
```
To inference on other videos, please specify the `--input_video` option and see more details in code.


## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@misc{li2023hdtrnet,
      title={HDTR-Net: A Real-Time High-Definition Teeth Restoration Network for Arbitrary Talking Face Generation Methods}, 
      author={Yongyuan Li and Xiuyuan Qin and Chao Liang and Mingqiang Wei},
      year={2023},
      eprint={2309.07495},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
},
```
