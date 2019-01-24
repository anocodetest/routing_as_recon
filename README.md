The files of datasets should be in the directory: './data/'.

run script:

```
python train_cnn_baseline.py --date date --logdir cnn --dataset mnist --epoch 200 --num_gpus 4 --conv1_channel_num 64 --num_capsule_primary 8 --loss_type softmax --weight_reg True --padding SAME
```

to train cnn baseline on mnist.

```
python train_caps_net.py --date date --logdir capsnet --dataset smallNORB --epoch 200 --primary_routing True --num_gpus 4 --iter_routing 2 --conv1_channel_num 64 --num_capsule_primary 8 --padding SAME
```

to train capsnet on smalNORB.

```
python train_em_routing.py --date date --logdir em --dataset SVHN --epoch 200 --num_gpus 4 --iter_routing 2 --conv1_channel_num 64 --num_capsule_primary 8 --padding SAME
```

to train em based routing on SVHN.

```
python train_routing_as_recon.py --date date --logdir rr --dataset cifar10 --epoch 200 --num_gpus 4 --conv1_channel_num 64 --num_capsule_primary 8 --iter_routing 2 --padding SAME
```

to train our proposed routing method on CIFAR10.

```
python train_wrn.py --date date --logdir wrn --dataset SVHN --num_res_block 2 --num_gpus 4 --batch_size 100 --epoch 240 --loss_type softmax --weight_reg True --learning_rate_decrease stage --init_learning_rate 0.1 --dropout_rate 0.4 --widen_factor 8 --decay_prop_0 0.4 --decay_prop_1 0.6 --decay_factor 0.1
```

to train WideResNet on SVHN.

```
python train_rr_wrn.py --date date --logdir rr_wrn --dataset SVHN --num_res_block 2 --num_gpus 4 --batch_size 100 --epoch 240 --loss_type softmax --weight_reg True --learning_rate_decrease stage --init_learning_rate 0.1 --dropout_rate 0.4 --widen_factor 4 --routing_widen_factor 1 --decay_prop_0 0.4 --decay_prop_1 0.6 --decay_factor 0.1
```

to train the hybrid model of WideResNet and routing as reconstruction on SVHN.

Records will be in the directory: './logdir/'

Requirements:
* Python 3.6
* TensorFlow 1.5.0