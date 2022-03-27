
# Runner commands for Imagenet

CUDA='0,1,2,3,4,5,6,7' #'0,1,2,3' # '0,1'
#DATA='/mnt/active/datasets/imagenet/' # 
DATA='/mnt/disks/data-disk/datasets/imagenet-1000/'
#DATA='/projectnb/datascigrp/anilkag/Imagenet-1000/data/'

ema_decay=0.999 #0.9999
B=256 #384 
LR=0.192  #0.1 
E=1 #450

#CUDA_VISIBLE_DEVICES=$CUDA ./distributed_train.sh 8 $DATA --model global_efficientnet_b0 -b $B \
#        --sched step --epochs $E --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 \
#        --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay $ema_decay \
#        --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr  $LR 


ema_decay=0.999 #0.9999
B=256 #384 
LR=0.192  #0.1 
E=1 #600

#CUDA_VISIBLE_DEVICES=$CUDA ./distributed_train.sh 8 $DATA --model global_mobilenetv3_large_100 -b $B \
#        --sched step --epochs $E --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 8 \
#        --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay $ema_decay \
#        --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr $LR  --lr-noise 0.42 0.9 #--resume $model_path  --no-resume-opt  --start-epoch 0 #--amp #--resume $model_path

ema_decay=0.999 #0.9999
B=128 #384 
LR=0.4  #0.1 
E=1 #200

CUDA_VISIBLE_DEVICES=$CUDA  ./distributed_train.sh 8 $DATA --model  global_mobilenetv2_100 -b $B \
       	--sched step --epochs $E --decay-epochs 2.4 --decay-rate .97 \
	--opt rmsproptf --opt-eps .001 -j 12 --warmup-lr 1e-6 --weight-decay 1e-5 \
	--drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay $ema_decay  \
	--aa rand-m9-mstd0.5 --remode pixel --reprob 0.05 --lr $LR #--resume $model_path  --start-epoch 0 --no-resume-opt 




