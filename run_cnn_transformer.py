import os,json,random,warnings
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt,seaborn as sns
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
ROOT=Path('/home/ubuntu/.cache/kagglehub/datasets/allandclive/chicken-disease-1/versions/1'); OUT=Path('/home/ubuntu/chicken_study/outputs'); OUT.mkdir(exist_ok=True)
IMG=ROOT/'Train'; df=pd.read_csv(ROOT/'train_data.csv'); classes=['Coccidiosis','Healthy','New Castle Disease','Salmonella']; c2i={c:i for i,c in enumerate(classes)};df['path']=df.images.map(lambda x:str(IMG/x));df['y']=df.label.map(c2i)
train_df,temp=train_test_split(df,test_size=.30,stratify=df.y,random_state=SEED);val_df,test_df=train_test_split(temp,test_size=.50,stratify=temp.y,random_state=SEED)
SIZE=128
train_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(p=.15),transforms.RandomRotation(25),transforms.ColorJitter(.25,.25,.25,.05),transforms.RandomResizedCrop(SIZE,scale=(.8,1.0)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225]),transforms.RandomErasing(p=.25,scale=(.02,.12),ratio=(.3,3.3))])
eval_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class DS(Dataset):
 def __init__(self,d,tf):self.d=d.reset_index(drop=True);self.tf=tf
 def __len__(self):return len(self.d)
 def __getitem__(self,i):
  r=self.d.iloc[i];return self.tf(Image.open(r.path).convert('RGB')),int(r.y)
tr=DataLoader(DS(train_df,train_tf),64,True,num_workers=2);va=DataLoader(DS(val_df,eval_tf),64,False,num_workers=2);te=DataLoader(DS(test_df,eval_tf),64,False,num_workers=2)
class CNNTransformer(nn.Module):
 def __init__(self,num_classes=4):
  super().__init__();self.cnn=nn.Sequential(nn.Conv2d(3,64,3,2,1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.GELU(),nn.Conv2d(128,192,3,2,1),nn.BatchNorm2d(192),nn.GELU(),nn.Conv2d(192,256,3,2,1),nn.BatchNorm2d(256),nn.GELU(),nn.Dropout2d(.15));self.pos=nn.Parameter(torch.zeros(1,64,256));layer=nn.TransformerEncoderLayer(d_model=256,nhead=8,dim_feedforward=512,dropout=.30,activation='gelu',batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,2);self.norm=nn.LayerNorm(256);self.head=nn.Sequential(nn.Dropout(.45),nn.Linear(256,num_classes))
 def forward(self,x):
  z=self.cnn(x);z=z.flatten(2).transpose(1,2);z=z+self.pos[:,:z.size(1)];z=self.encoder(z);z=self.norm(z.mean(1));return self.head(z)

def epoch(m,loader,opt=None):
 m.train(opt is not None);lossfn=nn.CrossEntropyLoss(label_smoothing=.1);tot=cor=ls=0;ys=[];ps=[]
 for x,y in loader:
  if opt:opt.zero_grad()
  z=m(x);loss=lossfn(z,y)
  if opt:loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
  pr=z.argmax(1);tot+=len(y);cor+=(pr==y).sum().item();ls+=loss.item()*len(y);ys+=y.tolist();ps+=pr.tolist()
 return ls/tot,cor/tot,ys,ps
m=CNNTransformer();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-3);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);h={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]};best=(-1,None)
for ep in range(30):
 tl,ta,_,_=epoch(m,tr,opt);vl,va_,_,_=epoch(m,va);sched.step();h['train_loss'].append(tl);h['val_loss'].append(vl);h['train_acc'].append(ta);h['val_acc'].append(va_)
 if va_>best[0]:best=(va_,{k:v.detach().clone() for k,v in m.state_dict().items()})
 print(ep+1,round(ta,4),round(va_,4),flush=True)
m.load_state_dict(best[1]);_,_,ys,ps=epoch(m,te);rep=classification_report(ys,ps,target_names=classes,output_dict=True,zero_division=0);cm=confusion_matrix(ys,ps)
torch.save({'model':m.state_dict(),'name':'CNNTransformer'},OUT/'CNNTransformer.pt');pd.DataFrame(rep).T.to_csv(OUT/'CNNTransformer_classification_report.csv');np.savetxt(OUT/'CNNTransformer_confusion_matrix.csv',cm,fmt='%d',delimiter=',')
json.dump({'CNNTransformer':h,'report':rep,'cm':cm.tolist(),'best_val_accuracy':best[0],'test_accuracy':rep['accuracy']},open(OUT/'CNNTransformer_results.json','w'),indent=2)
sns.set_theme(style='whitegrid');fig,ax=plt.subplots(1,2,figsize=(14,5));e=np.arange(1,31);ax[0].plot(e,h['train_acc'],label='Training');ax[0].plot(e,h['val_acc'],label='Validation');ax[0].set_title('CNN–Transformer accuracy');ax[0].set_xlabel('Epoch');ax[0].set_ylabel('Accuracy');ax[0].legend();ax[1].plot(e,h['train_loss'],label='Training');ax[1].plot(e,h['val_loss'],label='Validation');ax[1].set_title('CNN–Transformer loss');ax[1].set_xlabel('Epoch');ax[1].set_ylabel('Cross-entropy loss');ax[1].legend();fig.tight_layout();fig.savefig(OUT/'individual_model_figures/CNNTransformer_accuracy_loss.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6));sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes,ax=ax);ax.set_title('CNN–Transformer confusion matrix');ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(OUT/'individual_model_figures/CNNTransformer_confusion_matrix.png',dpi=220);plt.close(fig)
repdf=pd.DataFrame(rep).T.loc[classes,['precision','recall','f1-score']];fig,ax=plt.subplots(figsize=(8,5));sns.heatmap(repdf,annot=True,fmt='.3f',vmin=.7,vmax=1,cmap='YlGnBu',ax=ax);ax.set_title('CNN–Transformer classification report');fig.tight_layout();fig.savefig(OUT/'individual_model_figures/CNNTransformer_classification_report.png',dpi=220);plt.close(fig)
print('DONE',best[0],rep['accuracy'])
