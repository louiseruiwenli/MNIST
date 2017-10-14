####依赖
* `tensorflow`
* `keras`
* `numpy`
* `h5py`
* `bottle`
####测试环境
Mac OS X 10.11.6 with Python 3.4.3
####文件
* `MNIST.py`用于训练CNN模型并保存至`MNIST_model.h5`
* `MNIST_predict.py`用于读取并使用模型来预测数据
* `NumberDetection.py`用于调用摄像头来识别数字
* `Server.py`用于调用网页服务器实现模型的实时演示