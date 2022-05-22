Run the code with the following example commands:<br>
###  Uniform Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --schedule 40 80 --start_prune 40 --epochs 120
```
###  Class Dependent Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.4 --schedule 40 80 --start_prune 40 --epochs 120
```
###  Uniform Noise with noise rate 0.4 on CIFAR-100

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4 --schedule 80 120 --start_prune 80 --epochs 150
```
###  Class Dependent Noise with noise rate 0.4 on CIFAR-100

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.4 --schedule 80 120 --start_prune 80 --epochs 150
