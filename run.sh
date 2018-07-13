# pretrain
#CUDA_VISIBLE_DEVICES=0 python main.py --name=model_reconstruct_2d --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.00001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model_reconstruct_2d --is_train=True --learning_rate=0.000001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

CUDA_VISIBLE_DEVICES=0 python main.py --name=model_reconstruct_2d --batch_size=1 --gf_dim=32 --is_sup_train=False --is_train=False

#finetune
#rm -rf checkpoint/finetune_real/
#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune_real --batch_size=1 --model_dir=checkpoint/model0 --gf_dim=32 --is_sup_train=False --is_train=True --learning_rate=0.000001 --key_loss=True --max_iter=10000  #--silh_loss=True
#
#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune_real --batch_size=1 --model_dir=checkpoint/finetune_real --gf_dim=32 --is_sup_train=False --is_train=False
