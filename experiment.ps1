
########################################## icarl using vit lite ####################################################################
#python -u train.py --epochs 100 --project_name icarlonvit --feature_extractor vit_lite --online --batch_size 128 --learning_rate 0.1


########################################## icarl using cct7 ####################################################################
python -u train.py --epochs 2 --project_name icarloncct7_h --feature_extractor cct7_h --batch_size 32 --learning_rate 0.001 --online
