# GPU-FHE

- 代码目前需要用python3.8版本，3.10版本context类的初始化会有问题
- 本工程默认 存放随机数的是ax，存放mx数据的是bx，
    - 如果存在数组tmp[2]，那么 tmp[0]对应ax，tmp[1]对应bx 【这个存放方式与openfhe相反，但是与yhh早期c++工程一致】
- arithmetic.py中 以int结尾的函数表示输入或者输出涉及128位计算，以mod结尾的至多在内部涉及超过64位计算
- 以ct结尾的函数表示对密文两个分量做计算，否则对单个分量操作。
- 本工程keyswitch cuda算子仅支持logN >13 的数据.

---
TNT/GPU-FHE 开发指南

- 克隆指定仓库
  - git clone --recursive -b xx-branch https://...
- 用下面的命令编译
  - USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python3 setup.py
    develop --install-dir=~/torch/
- 运行示例程序
  - python3 dev.py

---

- ./torch/fhe/functional.py 写前端
  - i.e. 把 ./aten/src/ATen/native/native_functions.yaml 文件中的后端接口再封装一层
- aten/src/ATen/native/fhe/cuda/arithmetic.cu 中写基础算子
  - 在 namespace fhe 中写cuda核函数
  - 再写一个template调用上述核函数（负责dispatch）
  - namespace at::native 中返回 Tensor 的函数调用上述template 函数