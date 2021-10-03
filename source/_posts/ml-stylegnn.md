---
title: Using a pretrained model to generate fake faces
layout: MachineLearning Deepfake StyleGan
---

This article is to record my experience of 4 hours struggling with an easy script and hope it can help engineers like me to run this pre-trained model. Actually it is easy by the way, but I found there are so many engineers were searching these same problems again and again, but I have not found the answers online.

So here I am!

## The github project

My plan is to run a StyleGAN model to generate some fake faces and I find this owsome [github repo](https://github.com/NVlabs/stylegan). According to the demo code, this model can automatically generate a fake face without any other user data, which perfectly meets my needs. Not to mention that it has perfect paper support and simple API calls. After comparing several projects, I think this is the model I am looking for.

Since I don't have relevant hardware facilities locally (the GPU and CUDA), I planned to run the model on colab, which did not seem difficult. However, I ran into some problems.

Next, I will start with the first problem I encountered, and gradually explain how I discovered the problem, looked for a solution, and finally solved the problem. If you are just looking for a perfect colab running solution, you can directly access my [colab file](https://colab.research.google.com/drive/1O2iA3M6AM1rTzdvozIa8wp0moCoeTYk6?usp=sharing) and run it smoothly.

## Tensorflow version

I copied the demo from github first, installed the dependencies and run it. The first error is as follows.

```python
Error : module 'tensorflow' has no attribute 'Dimension' 
```

And thanks to google, I found the [solution](https://github.com/cedricoeldorf/ConditionalStyleGAN/issues/3) easily by just reinstalling tensorflow v1.15.2.

## Colab download error
As expected, I immediately encountered the next error. :)

```
Colab Google Drive quota exceeded
```

Copy the error into the google search box, the first result is the issue from colab toolkit github. After reading the code and the issue page, the solution quickly come into my mind. The error happened because the model file is a little large and google colab thinks that I have exceeded the download limit.

The solution is very simple. I download the file and upload it to google drive, so the model can be loaded locally from the drive, and colab does not think this is counted in the download.

The previous code.

```python
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
```

The Modified code.

```python
    url = './models/karras2019stylegan-ffhq-1024x1024.pkl' # karras2019stylegan-ffhq-1024x1024.pkl    
    _G, _D, Gs = pickle.load(open(url, 'rb'))
```

## GPU & CUDA

After solving the problem above, the next error arrives immediately of course. It looks like this.

```
The requested device appears to be a GPU, but CUDA is not enabled.
```

I am pretty new to colab and the first idea came to me is to make sure my GPU configuration of colab. But it looked fine. And I thought CUDA mush be the problem. Following the env requirements on the README.md, I tried to install CUDA 9.0 instead but it did not work. Then I thought this might be a compatibility issue between tensorflow and CUDA. So I tried more different versions. Not surprisingly, the error was still being reported.

Desperate, I read the error report carefully. It was mentioned in the error that tensorflow tried to find the GPU, but did not find a runnable one.

```
Cannot assign a device for operation Gs_4/_Run/Gs/latents_in: node Gs_4/_Run/Gs/latents_in (defined at /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/framework/ops.py:1748)  was explicitly assigned to /device:GPU:0 but available devices are [ /job:localhost/replica:0/task:0/device:CPU:0, /job:localhost/replica:0/task:0/device:XLA_CPU:0 ]. Make sure the device specification refers to a valid device. The requested device appears to be a GPU, but CUDA is not enabled.
```

So in colab environment, is the GPU really available? I googled the official [document of tensorflow](https://www.tensorflow.org/guide/gpu): "how to confirm the GPU situation" and get my answer. 

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

And that really save my life! It turns out that my tensorflow cannot access an available GPU. This is the reason! After knowing the reason for the error, everything went well. I quickly figured out that this was because I reinstalled tensorflow. The correct approach should be to modify the tensorflow version as follows.

```python
%tensorflow_version 1.x
```

This is what I actually need! I quickly delete those code trying to change CUDA version or python version. And the script was still fine.

The reason why we need to do this can be found in this [document](https://colab.research.google.com/notebooks/tensorflow_version.ipynb). And I would copy the points as follows.

```
Avoid Using pip install with GPUs and TPUs

We recommend against using pip install to specify a particular TensorFlow version for both GPU and TPU backends. Colab builds TensorFlow from source to ensure compatibility with our fleet of accelerators. Versions of TensorFlow fetched from PyPI by pip may suffer from performance problems or may not work at all.
```

## The end

At the end I would love to share some interesting knowledge I have learned in these four hours. A better understanding of colab will help us find problems faster.

* When running Notebook, colab will create a virtual machine environment, you can basically use it as an ubuntu machine.

* If there are a lot of files that need to be loaded from google drive, such as an image data set, the best pratice is to read a zip package from the drive and decompress it in the running virtual machine. This will greatly speed up the data reading speed.

* It is really important to read the error report carefully and analyze it :).

Actually there are some minor errors reported during the process, but they are not included in this article cos I believe that you are smart enough to solve them quickly and happily. :) ~

Thank you for reading and wish you a nice day. Bye~~

