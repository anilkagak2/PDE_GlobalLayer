


CUDA='0'
DATA=./data/
EPOCHS=160

# Resnet baseline
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_resnet --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -e $EPOCHS

# Neural-ODE baseline
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_odenet --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -e $EPOCHS

# Resnet-Global
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. -wd 1e-4 --non_linear -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. --non_linear -e $EPOCHS
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. --non_linear -e $EPOCHS


