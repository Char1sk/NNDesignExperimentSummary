# 神经网络课设II

## 项目代码说明

- 本仓库代码改自于[Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment)，忘记fork了
- 在此基础上，添加了confs配置文件，添加了results文件夹记录结果，添加了数个ipynb
- 实验基于[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)的官方数据集
- 相应实验的执行方式放在了Experiments.ipynb中，主要运行train和search脚本
- search脚本实际由于github库版本等原因，实际上无法运行

## 实验结果记录

| Num  |      Method      |       Model       | Search Dataset |   Note   |
| :--: | :--------------: | :---------------: | :------------: | :------: |
|      |                  |                   |                |          |
| 1.1  |      Plain       | Wide-ResNet-28-10 | CIFAR-10(4000) | Searched |
| 1.2  |   AutoAugment    | Wide-ResNet-28-10 | CIFAR-10(4000) | Searched |
| 1.3  | Fast AutoAugment | Wide-ResNet-28-10 | CIFAR-10(4000) | Searched |
|      |                  |                   |                |          |
| 2.1  |      Plain       | Wide-ResNet-40-2  | CIFAR-10(4000) | Searched |
| 2.2  |   AutoAugment    | Wide-ResNet-40-2  | CIFAR-10(4000) | Searched |
| 2.3  | Fast AutoAugment | Wide-ResNet-40-2  | CIFAR-10(4000) | Searched |
|      |                  |                   |                |          |
| 3.1  | Fast AutoAugment | Wide-ResNet-28-10 | CIFAR-10(4000) |  Search  |
| 3.2  | Fast AutoAugment | Wide-ResNet-28-10 | CIFAR-10(1000) |  Search  |
|      |                  |                   |                |          |
| 4.1  |      Plain       | Wide-ResNet-28-10 | CIFAR-10(4000) | Searched |
| 4.2  | Fast AutoAugment | Wide-ResNet-28-10 | CIFAR-10(4000) | Searched |
|      |                  |                   |                |          |


| Num  | Train Loss | Test Loss | Train Acc | Test Acc |
| :--: | :--------: | :-------: | :-------: | :------: |
|      |            |           |           |          |
| 1.1  |   0.2837   |  0.1452   |  0.9781   |  0.9583  |
| 1.2  |   0.4198   |  0.1247   |  0.9389   |  0.9575  |
| 1.3  |   0.4439   |  0.1267   |  0.9340   |  0.9572  |
|      |            |           |           |          |
| 2.1  |   0.2840   |  0.2031   |  0.9617   |  0.9410  |
| 2.2  |   0.4363   |  0.1676   |  0.9116   |  0.9431  |
| 2.3  |   0.4590   |  0.1730   |  0.9045   |  0.9424  |
|      |            |           |           |          |
| 3.1  |   0.4439   |  0.1267   |  0.9340   |  0.9572  |
| 3.2  |     -      |    -      |    -      |    -     |
|      |            |           |           |          |
| 4.1  |   0.1726   |  0.1350   |  0.9960   |  0.9649  |
| 4.2  |   0.2794   |  0.0918   |  0.9774   |  0.9724  |
|      |            |           |           |          |
