# 使用 CNN 对 CIFAR10 数据集进行分类

使用不同方法分类

已实现

- 普通 CNN

  89.63%

- Mixup TTA Ensemble

  91.05%

TODO：

- 迁移学习

- 知识蒸馏

- 半监督学习


## 说明

每个子目录下放置不同的实现，不同实现包括训练用到脚本文件和生成的目录

脚本文件

- `main.py` 运行即可开始训练
- `helper.py` 一般包含和该实现有关的函数和类
- `options.yaml` 训练时的各种参数设置
- `visualize.py` 与训练无关，训练完后进行可视化

目录

- `{project}/models/` 训练过程中保存的模型
- `{project}/output/` 生成的图像
- `{project}/log/` 日志A（包括 TensorBoard 日志和文本日志）
