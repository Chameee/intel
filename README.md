# intel

## 快速开始

### 拉取代码到本地
```
git clone git@github.com:Chameee/intel.git
```

### cmake 一下

```
cd intel/
cmake .
```

### 编译得到可执行文件

```
make
```

### 运行程序
```
please type the batch size: 
```
指定测试数据的 batch size, Exp: 2

```
please type the length of the 1D Tensor : 
```
指定测试一维数据的长度, Exp: 8

```
please type the rows of the input 2D Tensor: 
```
指定测试二维数据的行数 (即图像高度), Exp: 256

```
please type the columns of the input 2D Tensor: 
```
指定测试二维数据的列数 (即图像宽度), Exp: 256


1D和2D条件下的测试结果会分别保存在 **result_1d.txt** 和 **result_2d.txt** 中


### 测试说明

1. 测试数据
-  一维均匀分布随机数
-  二维均匀分布随机数

2. 测试卷积核
是一个均值化的核，可以自己在main函数里改

