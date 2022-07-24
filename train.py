from image_loader import *
from OGLA import OGLA
import torch
from torchvision import transforms
import joint_transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np
import torch.nn as nn
from misc import AvgMeter, check_mkdir
from torch.autograd import Function

cudnn.benchmark = True

torch.cuda.set_device(0)

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((256, 256))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((256, 256))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(training_root, joint_transform, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=12, num_workers=0, shuffle=True)


bce= nn.BCEWithLogitsLoss().cuda()
remove = torch.nn.L1Loss().cuda()


def wbce(pred, gt):
    pos = torch.eq(gt, 255).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return nn.functional.binary_cross_entropy_with_logits(pred, gt, weights)

def train():

    net = OGLA().cuda().train()
    optimizerd = torch.optim.Adamax([{'params': net.parameters()}], lr=0.0005)

    train_loss_global_record, loss_fuse_record, loss1_record = AvgMeter(), AvgMeter(), AvgMeter()
    loss2_record, loss3_record, loss4_record = AvgMeter(), AvgMeter(), AvgMeter()
    loss_global_record = AvgMeter()
    min=2000

    for epoch in range(100):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels= data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizerd.zero_grad()
            predict4_l2h, predict3_l2h, predict2_l2h, predict1_l2h, \
            predict_global, predict_fusion = net(inputs)            
            loss_global= bce(predict_global, labels)
            loss_fusion=bce(predict_fusion, labels)
            loss_1=bce(predict1_l2h, labels)
            loss_2=bce(predict2_l2h, labels)
            loss_3=bce(predict3_l2h, labels)
            loss_4=bce(predict4_l2h, labels)            
            loss=loss_global+loss_fusion+loss_1+loss_2+loss_3+loss_4            
            loss.backward()
            optimizerd.step()            
            loss_global_record.update(loss.data, batch_size)            
            train_loss_global_record.update(loss_global.data, batch_size)
            loss_fuse_record.update(loss_fusion.data, batch_size)
            loss1_record.update(loss_1.data, batch_size)
            loss2_record.update(loss_2.data, batch_size)
            loss3_record.update(loss_3.data, batch_size)
            loss4_record.update(loss_4.data, batch_size)

            epoch_loss += loss.item()

            print('Epoch: %d |iter:%d| loss_global_record: %.5f | train_loss_global_record: %.5f| loss_fuse_record: %.5f |' \
                  'loss1_record: %.5f | loss2_record: %.5f | loss3_record: %.5f | loss4_record: %.5f ' % \
                  (epoch, i, loss_global_record.avg,train_loss_global_record.avg,loss_fuse_record.avg,loss1_record.avg,
                   loss2_record.avg,loss3_record.avg,loss4_record.avg))                
        global min 
                
 
        if epoch_loss<=min:
            min=epoch_loss
            torch.save(net.state_dict(), './model/model_best.pth')

if __name__ == '__main__':

    train()
    
  