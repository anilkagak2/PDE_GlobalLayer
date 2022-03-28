


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
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. --non_linear -e $EPOCHS



#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear  --init_h0_h -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.2 --cDy 0.2 --non_linear  --init_h0_h -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 2. --cDy 2. --non_linear  --init_h0_h -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. --non_linear  --init_h0_h -e $EPOCHS



#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --old_style -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_res -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_res --use_f_for_g -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_res --use_f_for_g --old_style -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_res --old_style -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.2 --cDy 0.2 --non_linear  -e $EPOCHS

#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g --old_style -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g -e $EPOCHS

#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -wd 1e-4 --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -wd 1e-5 --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g -e $EPOCHS




#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -wd 8e-5 --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -wd 8e-5 --non_linear --use_f_for_g -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS

CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'BasicBlock' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'Bottleneck' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'DwConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'PwConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'FullConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_uv 'identity' 

CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'BasicBlock' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'Bottleneck' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'DwConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'PwConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'FullConv' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 1 --n1 16 -b 128 -wd 5e-5 --non_linear --use_f_for_g -e $EPOCHS --custom_dxy 'identity' 



#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128 -wd 3e-5 --constant_Dxy --cDx 0.5 --cDy 0.5 --non_linear --use_f_for_g -e $EPOCHS

#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 2. --cDy 2. --non_linear  -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model m_global --dataset MNIST-10 -m 0  --K 5 --n1 16 -b 128  --constant_Dxy --cDx 1. --cDy 1. --non_linear  -e $EPOCHS
