import torch
import torch.nn as nn
from ResNet.resnext101_regular import ResNeXt101
import torch.nn.functional as F
resnext_101_32_path = './model/resnet101.pth'
from torchsummary import summary


class ConvG(nn.Module):
    def __init__(self,in_dim=256, dim=512,kernel_size=3):
        super().__init__()
        self.in_dim=in_dim
        self.dim=dim
        self.kernel_size=kernel_size        
        self.conv=nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )
        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )

    def forward(self, x):
        
        x=self.conv(x)
        bs,c,h,w=x.shape       
        k1=self.key_embed(x)         
        v=self.value_embed(x).view(bs,c,-1) 
        y=torch.cat([k1,x],dim=1)         
        att=self.attention_embed(y) 
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w) 
        att=att.mean(2,keepdim=False).view(bs,c,-1) 
        k2=F.softmax(att,dim=-1)*v 
        k2=k2.view(bs,c,h,w)     
        out=k1+k2
        return out 

class convZ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convZ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)

class convZ_pool(nn.Module):
    def __init__(self, in_ch):
        super(convZ_pool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)
    
    
class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)
        return y
    
class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x 
    
        
class OGLA(nn.Module):
    def __init__(self):
        super(OGLA, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0   
        self.layer1 = resnext.layer1  
        self.layer2 = resnext.layer2  
        self.layer3 = resnext.layer3          
        self.conv1 = ConvG(3, 64) 
        self.pool1 = convZ_pool(64)  
        self.pool1_1 = convZ_pool(64)
        self.pool1_2 = convZ_pool(64)
        self.pool1_3 = convZ_pool(64)        
        self.conv2 = ConvG(64, 128)
        self.pool2 = convZ_pool(128)
        self.pool2_1 = convZ_pool(128)
        self.pool2_2 = convZ_pool(128)        
        self.conv3 = ConvG(128, 256)
        self.pool3 = convZ_pool(256)
        self.pool3_1 = convZ_pool(256)        
        self.conv4 = ConvG(256, 512) 
        self.pool4 = convZ_pool(512)
        self.conv5 = ConvG(512, 1024)         
        self.att3 = LayerConv(960, 512, 1, 1, 0, False)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  
        self.conv6 = convZ(1024, 512)                               
        self.att2 = LayerConv(448, 256, 1, 1, 0, False)        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)  
        self.conv7 = convZ(512, 256)                                  
        self.att1 = LayerConv(192, 128, 1, 1, 0, False)        
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)  
        self.conv8 = convZ(256, 128)                                 
        self.att4 = LayerConv(1984, 1024, 1, 1, 0, False)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)   
        self.conv9 = convZ(128, 64)                            
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)        
        self.conv11 = nn.Conv2d(32, 1, 1)                   
        self.convx1=nn.Conv2d(128, 64, 3, padding=1)      
        self.convx2 = nn.Conv2d(512, 256, 3, padding=1)     
        self.convx3 = nn.Conv2d(1024, 512, 3, padding=1)    
        self.convx4 = nn.Conv2d(2048, 1024, 3, padding=1)  
        self.layer4_conv3 = LayerConv(1024, 32, 1, 1, 0, False)
        self.layer3_conv3 = LayerConv(512, 32, 1, 1, 0, False)
        self.layer2_conv3 = LayerConv(256, 32, 1, 1, 0, False)      
        self.relu = nn.ReLU()
        self.global_conv = LayerConv(160, 32, 1, 1, 0, True)
        self.layer4_predict = Predict(32, 1, 1)
        self.layer3_predict_ori = Predict(32, 1, 1)
        self.layer3_predict = Predict(2, 1, 1)
        self.layer2_predict_ori = Predict(32, 1, 1)
        self.layer2_predict = Predict(3, 1, 1)
        self.layer1_predict_ori = Predict(64, 1, 1)
        self.layer1_predict = Predict(4, 1, 1)
        self.layer0_predict_ori = Predict(32, 1, 1)
        self.layer0_predict = Predict(5, 1, 1)
        self.global_predict = Predict(32, 1, 1)
        self.fusion_predict = Predict(5, 1, 1)
    def forward(self, x):
        layer0 = self.layer0(x)      
        layer1 = self.layer1(layer0) 
        layer2 = self.layer2(layer1)  
        layer3 = self.layer3(layer2)          
        layer3_output=layer3
        layer3_output = F.interpolate(layer3_output, size=(256,256), mode='bilinear', align_corners=False)
        c1 = self.conv1(x)               
        p1 = self.pool1(c1)
        p1_1=self.pool1_1(p1)  
        p1_2=self.pool1_2(p1_1)
        p1_3=self.pool1_3(p1_2)
        t1 = torch.cat([p1,layer0], dim=1) 
        x1 = self.convx1(t1)              
        c2 = self.conv2(x1)                
        p2 = self.pool2(c2)
        p2_1=self.pool2_1(p2)
        p2_2=self.pool2_2(p2_1)        
        c3 = self.conv3(p2)
        t2 = torch.cat([c3, layer1], dim=1) 
        x2 = self.convx2(t2)                
        p3 = self.pool3(x2)
        p3_1=self.pool3_1(p3)        
        c4 = self.conv4(p3)                    
        t3 = torch.cat([c4, layer2], dim=1) 
        x3 = self.convx3(t3)                 
        p4 = self.pool4(x3)
        c5 = self.conv5(p4)             
        t4 = torch.cat([c5, layer3], dim=1) 
        x4 = self.convx4(t4)                 
        att4_1=torch.cat([x4,p4,p3_1,p2_2,p1_3],dim=1)
        att4_1=self.att4(att4_1)        
        up_6 = self.up6(att4_1)
        att3_1=torch.cat([x3,p3,p2_1,p1_2],dim=1)
        att3_1=self.att3(att3_1)        
        merge6 = torch.cat([att3_1, up_6], dim=1)
        layer4_conv3 = self.layer4_conv3(merge6)
        layer4_up = F.upsample(layer4_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer4_up = self.relu(layer4_up)        
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)        
        att2_1=torch.cat([x2,p2,p1_1],dim=1)
        att2_1=self.att2(att2_1)        
        merge7 = torch.cat([att2_1, up_7], dim=1)
        layer3_conv3 = self.layer3_conv3(merge7)
        layer3_up = F.upsample(layer3_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_up = self.relu(layer3_up)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)        
        att1_1=torch.cat([c2,p1],dim=1)
        att1_1=self.att1(att1_1)       
        merge8 = torch.cat([att1_1, up_8], dim=1)
        layer2_conv3 = self.layer2_conv3(merge8)
        layer2_up = F.upsample(layer2_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_up = self.relu(layer2_up)                
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)        
        merge9 = torch.cat([c1, up_9], dim=1)
        layer1_up = self.conv9(merge9)        
        global_concat = torch.cat((layer1_up, layer2_up, layer3_up, layer4_up), 1)
        global_conv = self.global_conv(global_concat)
        layer4_predict = self.layer4_predict(layer4_up)
        layer3_predict_ori = self.layer3_predict_ori(layer3_up)
        layer3_concat = torch.cat((layer3_predict_ori, layer4_predict), 1)
        layer3_predict = self.layer3_predict(layer3_concat)
        layer2_predict_ori = self.layer2_predict_ori(layer2_up)
        layer2_concat = torch.cat((layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer2_predict = self.layer2_predict(layer2_concat)        
        layer1_predict_ori = self.layer1_predict_ori(layer1_up)
        layer1_concat = torch.cat((layer1_predict_ori, layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)###2,4,256,256
        layer1_predict = self.layer1_predict(layer1_concat)
        global_predict = self.global_predict(global_conv)
        fusion_concat = torch.cat((layer1_predict, layer2_predict, layer3_predict,
                                   layer4_predict, global_predict), 1)
        fusion_predict = self.fusion_predict(fusion_concat)      
        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict, global_predict, fusion_predict
        return nn.Sigmoid()(fusion_predict)
        
if __name__ == '__main__':
    
    
    x = torch.randn((2, 3, 256, 256))
    net = OGLA().cuda()
    summary(net, (3, 256, 256))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
