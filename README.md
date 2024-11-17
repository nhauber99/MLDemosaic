# MLDemosaic
(work in progress)

Python code for training, evaluating and running a machine learning based demosaicing model.
The concept behind the model architecture is to assume a fixed bayer pattern, use PixelUnshuffle to transform the values of each bayer patch into 4 channels at the same position, thus halfing the width. The resulting tensor is the input to 3 Residual-in-Residual Dense Blocks (from the [ESRGAN paper](https://arxiv.org/pdf/1809.00219)). A pixel shuffle operation upscales the image again followed by a convolution. This network should only predict the residual of the result obtained by bilinear interpolation.

![image](https://github.com/user-attachments/assets/dfe44193-daa5-4877-894a-81fc5987c92c)

As training data about 10.000 downscaled images of the the [OpenImages V7](https://storage.googleapis.com/openimages/web/index.html) dataset are used.

Preliminary results are very promising and the network seems to beat existing demosaicing algorithms like AHD or VNG by a large margin in terms of PSNR and also in terms of artifacts when doing visual comparisons of the result. The datasets used for testing are the [Kodak](https://r0k.us/graphics/kodak/) and [McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm) dataset. This does come at the cost of longer computation times, although the model runs at a very reasonable performance (40ms per megapixel on a RTX4070Ti). I will add more information soon.
