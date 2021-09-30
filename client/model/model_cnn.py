import torch
import torch.nn as nn
class CNN(nn.Module):
    
    def __init__(self,h,w,num_features):
        super(CNN,self).__init__()
        self.num_features = num_features
        self.layer1 = nn.Sequential(  
            nn.Conv2d(num_features, 16, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(16),  
            nn.ReLU(),	
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),	
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(h*w*2, 1)   
    def forward(self, x):
        x = x.reshape(len(x), -1, self.num_features)
        x = x.view((x.size(0),8,-1,x.size(2))).permute(0,3,1,2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
