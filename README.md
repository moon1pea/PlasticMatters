
# 训练数据集geniamge/sdv4/train  88G
# 测试数据集genimage/*/test   26G

# 下载AIDE项目原始权重 —— sd14_train.pth
https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4


# 运行命令

# 1、训练：
 ./scripts/train.sh   --data_path path/to/train_dataset   --eval_data_path path/to/test_dataset  --resume path/to/sd14_train.pth   --output_dir path/to/weights   --use_freq_sens_cbp True  --cbp_replacement_rate 1e-5   --freq_sens_lambda 0.5 --blr 1e-5

# 示例：使用下载好的上述数据集在sd14权重上finetuning
# 注意：数据集路径要写sdv4的上一级路径，即dataset/train/sdv4,要写train。
# 数据集格式应该为：sdv4
#                   |- 1_fake
#                   |- 0_real
 ./scripts/train.sh   --data_path dataset/train   --eval_data_path dataset/train  --resume checkpoints/sd14_train.pth   --output_dir results/sd14_finetune   --use_freq_sens_cbp True  --cbp_replacement_rate 1e-5   --freq_sens_lambda 0.5 --blr 1e-5


 # 2、评估
 ./scripts/eval.sh --data_path path/to/train_dataset --eval_data_path path/to/test_dataset --resume path/to/(训练后的权重) --eval True --output_dir path/to/result  --use_freq_sens_cbp True

# 示例：
./scripts/eval.sh --data_path dataset/train/sdv4 --eval_data_path dataset/test1 --resume results/sd14_blr_finetune/checkpoint-2.pth --eval True --output_dir results/sd14_finetune --use_freq_sens_cbp True
