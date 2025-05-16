# Convolutional LSTM Water Quality Prediction

水质监测浮标

## 代码运行环境 Requirements

- Python 3.12.10
- Nvidia GPU driver 576.40

Create Python virtual environment.

```powershell
python -m venv convlstm
```

Activate the virtual environment.

```powershell
.\convlstm\Scripts\Activate.ps1
```

Install PyTorch.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install dependencies.

```powershell
pip install keras matplotlib imageio ipython ipywidgets
```

Change the backend of Keras to PyTorch.

```powershell
code C:\Users\Moroshima\.keras\keras.json
```

```json
{
  "floatx": "float32",
  "epsilon": 1e-7,
  "backend": "torch",
  "image_data_format": "channels_last"
}
```

## 模型评估 Model Evaluation

回归预测模型

- 均方根误差 RMSE
- 决定系数/判定系数 R2/R-Squared

[On Powershell 7.5 preview 4, after `conda activate base`, `Env:_CE_CONDA` and `Env:_CE_M` appears again. · Issue #14292 · conda/conda](https://github.com/conda/conda/issues/14292)

## 参考文献 References

[Convlstm 时空预测（keras 框架、实战）\_convlstm3d 多步输出-CSDN 博客](https://blog.csdn.net/popofzk/article/details/106155925)

[时空预测 1-convlstm 中的 ecnoder-forecast\_表格化数据做 convlstm-CSDN 博客](https://blog.csdn.net/weixin_38812492/article/details/125276794)

[Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm/)

<https://www.cs.toronto.edu/~nitish/unsupervised_video/>

[Getting started with Keras](https://keras.io/getting_started/)

[Get Started - pytorch.org](https://pytorch.org/get-started/locally/)

### NOT IN USE

[keras-io/conv-lstm · Hugging Face](https://huggingface.co/keras-io/conv-lstm)

[conv-LSTM 解读：背景&介绍&优劣势&适用场景（附生成视频 seq 预测问题的 keras 代码） - 知乎](https://zhuanlan.zhihu.com/p/124106729)
