import wandb

from IncrementalTrainingApproach.iCaRL import iCaRLmodel
from models.ResNet import resnet18_cbam
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

numclass = 10
feature_extractor = resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0

args = dict(
    feature_extractor=resnet18_cbam(),
    img_size=32,
    batch_size=128,
    task_size=10,
    memory_size=2000,
    epochs=100,
    learning_rate=2.0,
)

model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
# model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

for i in range(10):
    args["task_id"] = i
    run = wandb.init(
        project="icarl1",
        reinit=True,
        config=args,
    )
    wandb.watch(model.model)
    model.beforeTrain()
    accuracy = model.train()
    model.afterTrain(accuracy)
    run.finish()
