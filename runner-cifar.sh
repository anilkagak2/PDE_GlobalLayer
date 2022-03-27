


CUDA='0'
DATA=./data/
EPOCHS=300

#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-10 -m 18 --n1 16 --n2 32 --n3 64  -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-100 -m 18 --n1 16 --n2 32 --n3 64   -e $EPOCHS


# Resnet32 (m=5) (460K, 70M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-10 -m 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o   -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-100 -m 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o   -e $EPOCHS

# Resnet32 Global-Diffusion (m=1) (170K, 15M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-10 -m 1 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e $EPOCHS  
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-100 -m 1 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e $EPOCHS  




# Resnet56 (m=9) (850K, 127M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-10 -m 9 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o   -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet --dataset CIFAR-100 -m 9 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o   -e $EPOCHS

# Resnet56 Global-Diffusion (m=2) (330K, 30M)
CUDA_VISIBLE_DEVICES='0' python train.py --data $DATA --model Resnet-Global --dataset CIFAR-10 -m 2 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 8e-5  --non_linear  -e $EPOCHS & 
CUDA_VISIBLE_DEVICES='1' python train.py --data $DATA --model Resnet-Global --dataset CIFAR-100 -m 2 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 8e-5 --non_linear  -e $EPOCHS &



# DenseNet-BC (800K, 300M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Densenet --dataset CIFAR-10 -m 1 --K 3 --n1 16 --n2 16 --n3 16 -b 64 -wd 1e-4  -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Densenet --dataset CIFAR-100 -m 1 --K 3 --n1 16 --n2 16 --n3 16 -b 64 -wd 1e-4  -e $EPOCHS

# DenseNet-BC Global Diffusion (526K, 168M)
CUDA_VISIBLE_DEVICES='6' python train.py --data $DATA --model Densenet-Global --dataset CIFAR-10 -m 2 --K 1 --n1 8 --n2 8 --n3 8 -b 64 -wd 1e-4  --non_linear  -e $EPOCHS &
CUDA_VISIBLE_DEVICES='7' python train.py --data $DATA --model Densenet-Global --dataset CIFAR-100 -m 2 --K 1 --n1 8 --n2 8 --n3 8 -b 64 -wd 1e-4 --non_linear   -e $EPOCHS &  
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Densenet-Global --dataset CIFAR-10 -m 2 --K 3 --n1 8 --n2 8 --n3 8 -b 64 -wd 1e-4  --non_linear  -e $EPOCHS 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Densenet-Global --dataset CIFAR-100 -m 2 --K 3 --n1 8 --n2 8 --n3 8 -b 64 -wd 1e-4  --non_linear   -e $EPOCHS  



# Wide-Resnet WRN-40-4 (9M, 1.3B)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet --dataset CIFAR-10 -m 6 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 1e-4 -wdt 4  -e $EPOCHS # &
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet --dataset CIFAR-100 -m 6 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 1e-4 -wdt 4  -e $EPOCHS # &

# Wide-Resnet WRN-40-4 Global Diffusion  (2.8M, 425M)
CUDA_VISIBLE_DEVICES='2' python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-10 -m 1 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 5e-5 -wdt 4    --non_linear  -e $EPOCHS  & 
CUDA_VISIBLE_DEVICES='3' python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-100 -m 1 --K 1 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 5e-5 -wdt 4    --non_linear  -e $EPOCHS  &
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-10 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 8e-5 -wdt 4     --non_linear  -e $EPOCHS 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-100 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 8e-5 -wdt 4  --non_linear  -e $EPOCHS 




# DARTS (3.3M, 539M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS --dataset CIFAR-10 -m 20 --K 1 --n1 36 -b 96 -wd 3e-4 -lr 0.025 -ct    -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS --dataset CIFAR-100 -m 20 --K 1 --n1 36 -b 96 -wd 3e-4 -lr 0.025 -ct    -e $EPOCHS

# DARTS Global Diffusion (783K, 213M)
CUDA_VISIBLE_DEVICES='4' python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-10 -m 6 --K 1 --n1 36 -b 96 -wd 8e-4 -lr 0.025 -ct   --non_linear  -e $EPOCHS &
CUDA_VISIBLE_DEVICES='5' python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-100 -m 6 --K 1 --n1 36 -b 96 -wd 8e-4 -lr 0.025 -ct  --non_linear  -e $EPOCHS & 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-10 -m 6 --K 5 --n1 36 -b 96 -wd 8e-4 -lr 0.025 -ct  --non_linear -e $EPOCHS
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-100 -m 6 --K 5 --n1 36 -b 96 -wd 8e-4 -lr 0.025 -ct  --non_linear  -e $EPOCHS 





#### Comparison with baseline at same FLOPs 
#### Baseline Models

# Resnet (13K, 3.4M)
#CUDA_VISIBLE_DEVICES=$CUDA  python train.py --data $DATA --model Resnet --dataset CIFAR-10 -m 5 -K 1 --n1 6 --n2 8 --n3 8 --n4 8 -b 32 -wd 1e-4  -e $EPOCHS  
#CUDA_VISIBLE_DEVICES=$CUDA '0' python train.py --data $DATA --model Resnet --dataset CIFAR-100 -m 3 -K 1 --n1 6 --n2 8 --n3 8 --n4 8 -b 32 -wd 1e-4  -e $EPOCHS  # &

# Resnet-Global (20K, 3.4M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-10 -m 1 -K 1 --n1 6 --n2 8 --n3 16 --n4 16 -b 32 -wd 1e-4  --non_linear  -e $EPOCHS  
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-100 -m 1 -K 1 --n1 6 --n2 8 --n3 16 --n4 16 -b 32 -wd 1e-4 --non_linear  -e $EPOCHS   # &


# Resnet-Global Diffusion (14K, 3.6M, 82.55 CIFAR-10)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-100 -m 1 -K 5 --n1 9 --n2 9 --n3 16 --n4 16 -b 32 -wd 2e-4  --non_linear  -e $EPOCHS   &
# Resnet-Global Diffusion (16K, 3.6M, 43.62 CIFAR-100)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model Resnet-Global --dataset CIFAR-10 -m 1 -K 1 --n1 10 --n2 9 --n3 16 --n4 16 -b 32 -wd 3e-5  --non_linear  -e $EPOCHS  &

# Wide-Resnet WRN-40-4 (22K, 9.8M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet --dataset CIFAR-10 -m 6 -K 1 --n1 2 --n2 2 --n3 2 --n4 2 -b 32 -wd 1e-4 -wdt 4    -e $EPOCHS   # &
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet --dataset CIFAR-100 -m 6 -K 1 --n1 2 --n2 2 --n3 2 --n4 2 -b 32 -wd 1e-4 -wdt 4   -e $EPOCHS   # &

# Wide-Resnet WRN-40-4 Global Diffusion  (22K, 8.9M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-10 -m 1 -K 1 --n1 3 --n2 4 --n3 3 --n4 3 -b 32 -wd 1e-4 -wdt 4  --non_linear  -e $EPOCHS  # &
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-100 -m 1 -K 1 --n1 3 --n2 4 --n3 3 --n4 3 -b 32 -wd 1e-4 -wdt 4 --non_linear  -e $EPOCHS  # &


# (23K, 8.7M) -- ( 85.5% CIFAR-10, 50.23% CIFAR-100 )
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-10 -m 1 -K 1 --n1 4 --n2 5 --n3 5 --n4 5 -b 32 -wd 1e-4 -wdt 4  --non_linear  -e $EPOCHS    # &
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA --model WideResnet-Global --dataset CIFAR-100 -m 1 -K 1 --n1 4 --n2 5 --n3 5 --n4 5 -b 32 -wd 1e-4 -wdt 4  --non_linear   -e $EPOCHS  # &

# DARTS (39K, 7.7M)
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS --dataset CIFAR-10 -m 20 -K 1 --n1 3 -b 96 -wd 1e-4 -lr 0.025 -ct   -e $EPOCHS 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS --dataset CIFAR-100 -m 20 -K 1 --n1 3 -b 96 -wd 3e-4 -lr 0.025 -ct   -e $EPOCHS 

# DARTS Global Diffusion (33K, 7.8M) 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-10 -m 6 -K 1 --n1 5 -b 96 -wd 1e-4 -lr 0.025 -ct   --non_linear  -e $EPOCHS 
#CUDA_VISIBLE_DEVICES=$CUDA python train.py --data $DATA  --model DARTS-Global --dataset CIFAR-100 -m 6 -K 1 --n1 5 -b 96 -wd 1e-4 -lr 0.025 -ct   --non_linear  -e $EPOCHS 
