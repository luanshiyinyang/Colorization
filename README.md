# Colorization


## 简介
- 本项目使用Keras复现论文[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)内容。
- 在Github上面找到的有质量的复现代码均为TensorFlow1.x，这对很多神经网络的新人是不友好的，且前面两者的代码可读性没有Keras强，所以使用Keras复现。


## 配置
本项目所需要的第三方包均在[requirements](/requirements.txt)中列出。使用下面命令安装。（**保证你在本项目根目录**）
- `pip install -r requirements.txt`


## 数据准备
访问云盘[链接](https://pan.baidu.com/s/1XtVUDiMFxeJoEKqVQiqUeQs)，下载数据集和预训练模型（提取码19p5）。



## 运行说明
- 训练
	- 用途
		- 本脚本用于模型的训练，训练数据要求以多个图片文件放置在根目录下的data/images/train中，脚本会自动划分验证集。
	- 命令行参数
		- **使用`python train.py -h`或者`python train.py --help`查看参数说明**
		- 使用`python train.py`将使用默认配置运行脚本
		- 参数说明
			- -p | --pretrained [预训练模型位置]
				- 使用此选项加载预训练模型，在此基础上训练
			- -s | --show [yes|no]
				- 是否显示训练过程
			- -m | --method [all|generator]
				- 是一次读入所有数据，还是多次读入
			- -b | --batch [2^n]
				- 训练批量大小
			- -e | --epochs [>=0]
				- 训练轮次
			- -l | --loss [ce|vce]
				- 使用哪种损失函数，可以使用交叉熵，或者再平衡的交叉熵
- test
	- 用途
		- 本脚本用于训练完成的模型的测试。将需要测试的彩色或者灰度图片转为灰度后上色。
	- 命令行参数
		- **使用`python test.py -h`或者`python test.py --help`查看参数说明**
		- 使用`python test.py`将使用默认配置运行脚本
		- 参数说明
			- -i | --input 输入图片的位置
				- 不指定将使用根目录下data/images/test文件夹作为默认位置。
			- -o | --output 输出图片位置
				- 不指定将根目录下results作为输出目录（不存在则创建）。


## 网络搭建
- 使用Keras Function API搭建（使用[Netron](https://lutzroeder.github.io/netron/)可视化模型构建结构）
- 模型概念图（摘自论文）
	- ![](/asset/structure.png)
- 实际模型
	- 由于设备限制，没有在imagenet全集上训练，而是挑选了部分风景子集，所以图片的数量不多，为了减少参数，最后两个block的卷积核数目均做了减半处理。


## 模型训练
如果在ImageNet上训练需要较长时间，但是效果较好，小数据集泛化能力较差。
