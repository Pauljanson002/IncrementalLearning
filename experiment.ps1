
########################################## icarl using vit lite ####################################################################
#python -u train.py --epochs 100 --project_name icarlonvit --feature_extractor vit_lite --online --batch_size 128 --learning_rate 0.1


########################################## icarl using cct7 ####################################################################
######## No augmentations ######################################################################################################
#python -u train.py --epochs 2 --project_name icarloncct7_h --feature_extractor cct7_h --batch_size 32 --learning_rate 0.001 --online

#########################################icarl using cct7 with different  learning rate#########################################
###################No augmentations ####################################################################################
#python -u train.py --epochs 100 --project_name icarloncct7_h_2 --feature_extractor cct7_h --batch_size 32 --learning_rate 2


########################################### icarl on cct7 with adamw and augmentations ########################################3
python -u train.py --epochs 6 --project_name icarloncct7_h_adam --feature_extractor cct7_h --batch_size 32 --learning_rate 0.0005