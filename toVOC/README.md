`caffe_train` 和 `cascade_train` 各自有可以把 `docs/车尾数据集.md` 所规定格式的 `trainset` 目录转换为 Caffe 和 `opencv_traincascade` 所需数据格式的脚本 `generate.py`, 以及训练模型的脚本 `train.py`; `scripts/evaluation` 下的测评脚本可以调用 `python` 下模块的 API 对 `testset` 进行测评；`regularize/regularize.py` 负责把标注文件转换为 `docs/车尾数据集.md` 所规定格式，同时一并检查样本的完整性。

均支持 Python 2.7 或 Python 3.4 以上的版本，但需要安装好 `python/requirements.txt` 所指定的第三方 Python 库。

## cascade_train 所要注意的

上游的 `opencv_traincascade` 命令对 negative samples 的要求是 "Negative samples are taken from arbitrary images. These images must not contain detected objects. ", 且用一个文本文件 `background.txt` 在每行列出图片的文件位置。`opencv_traincascade` 会用 `-bg` 参数接收这文本文件 `background.txt` 的路径，再读取每行所对应的图片，在上面用 `-w` 和 `-h` 所指定大小的 sliding window 扫描一遍，生成所有负样本。但现有车尾数据集所标注过的图片或帧几乎都有车即 "detected object", 所以不能直接放进 `background.txt`, 但是当前在有五万正样本的车尾数据集找出完全不包含 detected object 的图片或帧只有一千三多张，而且实践证明训练的效果很差。

于是我 hack 了 OpenCV, 重构了原本靠 sliding windows 扫描负样本 的源代码，不光大幅度改善了可读性，而且原本用来传入 `opencv_createsamples` 并生成 vec 文件的 `info.txt` 现在也可以作为 `-bg` 参数传给 `opencv_traincascade` 了。后者会依次读取每行所指定的图片，用 sliding windows 扫描负样本，但会自动过滤掉和正样本有重合的负样本。

hack 的源代码已打包成 `cascade_train/opencv3.patch`, 只需按 `cascade_train/patch.sh` 自行对 OpenCV 3.1.0 源代码打补丁，再自行编译安装即可，此外目前这补丁自然可能只适用于 OpenCV 3.1.
