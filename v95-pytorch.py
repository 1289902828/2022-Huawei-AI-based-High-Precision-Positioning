




#v95


import os
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed):
	torch.manual_seed(seed)  # 固定随机种子（CPU）
	if torch.cuda.is_available():  # 固定随机种子（GPU)
		torch.cuda.manual_seed(seed)  # 为当前GPU设置
		torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
	#np.random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_everything(43)





BATCH_SIZE = 20
LEARNING_RATE = 0.0001
TOTAL_EPOCHS = 1500
split_ratio = 0.1
change_learning_rate_epochs = 100

model_save = '/hy-tmp/model/modelSubmit_20220928_2_v95.pth'

DEVICE=torch.device("cpu")
if torch.cuda.is_available():
	DEVICE=torch.device("cuda:0")






class MyDataset(Dataset):
	def __init__(self, trainX, trainY, split_ratio, mode='train'):
		
		N = trainX.shape[0]
		
		TrainNum = int((N*(1-split_ratio)))
		
		if mode=='train':
			#self.x = trainX[:TrainNum].astype(np.float32)
			#self.y = trainY[:TrainNum].astype(np.float32)
			self.x = trainX.astype(np.float32)
			self.y = trainY.astype(np.float32)
		else:
			#self.x = trainX[TrainNum:].astype(np.float32)
			#self.y = trainY[TrainNum:].astype(np.float32)
			self.x = trainX.astype(np.float32)
			self.y = trainY.astype(np.float32)
		
		self.cut_length = 8
		
		self.len = len(self.y)
	
	def __len__(self):
		return self.len
	
	def __getitem__(self, idx):
		x = self.x[idx]
		y = self.y[idx]
		
		#数据增强cutout
		mask = np.ones((256,72,2), np.float32)
		for k in range(0,72):
			s1 = np.random.randint(256-self.cut_length)
			s2 = np.random.randint(256-self.cut_length)
			s3 = np.random.randint(256-self.cut_length)
			s4 = np.random.randint(256-self.cut_length)
			mask[s1:s1+self.cut_length,k,:] = 0
			mask[s2:s2+self.cut_length,k,:] = 0
			mask[s3:s3+self.cut_length,k,:] = 0
			mask[s4:s4+self.cut_length,k,:] = 0
		
		x = x * mask
		
		return (x, y)



class Model_2(nn.Module):
	def __init__(self, no_grad=True, infer_batchsize=5):
		super(Model_2, self).__init__()
		
		self.no_grad = no_grad
		
		self.infer_batchsize = infer_batchsize
		
		self.relu = nn.LeakyReLU(negative_slope=0)
		
		self.conv1 = nn.Conv2d(2, 256, kernel_size = (3,2), stride = (2,2), padding= (1,0))
		self.conv2 = nn.Conv2d(256, 512, kernel_size = (3,2), stride = (2,2), padding= (1,0))
		self.conv3 = nn.Conv2d(512, 768, kernel_size = (3,1), stride = (2,1), padding= (1,0))
		self.conv4 = nn.Conv2d(768, 256, kernel_size = (3,1), stride = (2,1), padding= (1,0))
		self.conv5 = nn.Conv2d(256, 512, kernel_size = (3,1), stride = (2,1), padding= (1,0))
		self.conv6 = nn.Conv2d(512, 768, kernel_size = (3,1), stride = (2,1), padding= (1,0))
		#256，512，768，256，512，768
		self.bn1 = nn.BatchNorm2d(256)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(768)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(512)
		self.bn6 = nn.BatchNorm2d(768)
		
		#self.pool  = nn.MaxPool2d(kernel_size = (2,1), stride = (2,1), padding = 0)
		self.avgpool  = nn.AvgPool2d(kernel_size = (2,1), stride = (1,1), padding = (1,0))
		
		self.Flatten = nn.Flatten()
		
		#self._dropout = nn.Dropout(0.5)
		
		self.fc_1  = nn.Linear(69120,2)
		#self.fc_2  = nn.Linear(1024,2)
	
	def forward_with_grad(self, x):
		
		x = x.permute(0,3,1,2)#[batch_size, 2, 256, 72]
		
		x = self.avgpool(x)
		x = self.avgpool(x)
		x = self.avgpool(x)
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		#x = self.pool(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		#x = self.pool(x)
		
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		#x = self.pool(x)
		
		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)
		
		x = self.conv5(x)
		x = self.bn5(x)
		x = self.relu(x)
		
		x = self.conv6(x)
		x = self.bn6(x)
		x = self.relu(x)
		#print(x.shape)
		#[bs,768,5,18]
		x = self.Flatten(x)
		
		out = self.fc_1(x)
		
		return out
		
	
	def forward_without_grad(self, x):
		with torch.no_grad():
			
			x = x.permute(0,3,1,2)#[batch_size, 2, 256, 72]
			
			x = self.avgpool(x)
			x = self.avgpool(x)
			x = self.avgpool(x)
			
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			#x = self.pool(x)
			
			x = self.conv2(x)
			x = self.bn2(x)
			x = self.relu(x)
			#x = self.pool(x)
			
			x = self.conv3(x)
			x = self.bn3(x)
			x = self.relu(x)
			#x = self.pool(x)
			
			x = self.conv4(x)
			x = self.bn4(x)
			x = self.relu(x)
			
			x = self.conv5(x)
			x = self.bn5(x)
			x = self.relu(x)
			
			x = self.conv6(x)
			x = self.bn6(x)
			x = self.relu(x)
			
			x = self.Flatten(x)
			
			out = self.fc_1(x)
			
			return out
	
	def forward(self, x):
		
		if self.no_grad:
			b, _, _, _ = x.shape
			mini_batch_outs = []
			for i in range(0, b, self.infer_batchsize):
				_out = self.forward_without_grad(x[i:i+self.infer_batchsize])
				#print(_out.shape)
				mini_batch_outs.append(_out)
			
			out = torch.cat(mini_batch_outs, axis=0)
		else:
			out = self.forward_with_grad(x)
			
		return out






if __name__ == '__main__':
	
	file_name1 = '/hy-tmp/data_unzip/data/Case_3_Training.npy'
	#print('The trainX dataset is : %s'%(file_name1))
	CIR = np.load(file_name1)
	trainX = CIR.transpose((2,1,3,0))
	
	file_name2 = '/hy-tmp/data_unzip/data/Case_3_Training_Label.npy'
	#print('The trainY dataset is : %s'%(file_name2))
	POS = np.load(file_name2)
	trainY = POS.transpose((1,0))
	
	
	#处理数据，提取有标签样本
	trainX = trainX[0:1000,:,:,:]
	
	
	model = Model_2(no_grad=False)
	model = model.to(DEVICE)
	print(model)
	
	train_dataset = MyDataset(trainX, trainY, split_ratio, mode='train')
	train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	val_dataset = MyDataset(trainX, trainY, split_ratio, mode='val')
	val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	
	criterion = nn.L1Loss().to(DEVICE)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	
	
	test_avg_min = 10000;
	print('开始训练------')
	for epoch in range(TOTAL_EPOCHS):
		
		model.train()
		
		#Training in this epoch  
		loss_avg = 0
		for i, (x, y) in enumerate(train_loader):
			x = x.float().to(DEVICE)
			y = y.float().to(DEVICE)
			
			# 清零
			optimizer.zero_grad()
			output = model(x)
			# 计算损失函数
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()
			
			loss_avg += loss.item() 
			
		loss_avg /= len(train_loader)
		
		#Testing in this epoch
		model.eval()
		test_avg = 0
		for i, (x, y) in enumerate(val_loader):
			x = x.float().to(DEVICE)
			y = y.float().to(DEVICE)
			with torch.no_grad():
				output = model(x)
				# 计算损失函数
				loss_test = criterion(output, y)
				test_avg += loss_test.item() 
		
		test_avg /= len(val_loader)
		
		if test_avg < test_avg_min:
			#print('Model saved!')
			test_avg_min = test_avg
			
			torch.save(model.state_dict(), model_save)
		
		print('Epoch : %d/%d, train_loss: %.4f, test_loss: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg, test_avg, test_avg_min))







