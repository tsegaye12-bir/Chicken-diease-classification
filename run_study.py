import os, json, random, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path('/home/ubuntu/.cache/kagglehub/datasets/allandclive/chicken-disease-1/versions/1')
OUT=Path('/home/ubuntu/chicken_study/outputs'); OUT.mkdir(parents=True,exist_ok=True)
IMG=ROOT/'Train'; csv=ROOT/'train_data.csv'
classes=['Coccidiosis','Healthy','New Castle Disease','Salmonella']; c2i={c:i for i,c in enumerate(classes)}
df=pd.read_csv(csv); df['path']=df.images.map(lambda x:str(IMG/x)); df['y']=df.label.map(c2i)
# verify all files
assert df.path.map(os.path.exists).all()
train_df, temp_df=train_test_split(df,test_size=.30,stratify=df.y,random_state=SEED)
val_df, test_df=train_test_split(temp_df,test_size=.50,stratify=temp_df.y,random_state=SEED)
for name,x in [('all',df),('train',train_df),('val',val_df),('test',test_df)]:
    x[['images','label']].to_csv(OUT/f'{name}_manifest.csv',index=False)
    print(name,len(x),x.label.value_counts().to_dict())

SIZE=128
train_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(20),transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
eval_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class DS(Dataset):
 def __init__(self,d,tf): self.d=d.reset_index(drop=True); self.tf=tf
 def __len__(self): return len(self.d)
 def __getitem__(self,i):
  r=self.d.iloc[i]; im=Image.open(r.path).convert('RGB'); return self.tf(im),int(r.y)
train=DS(train_df,train_tf); val=DS(val_df,eval_tf); test=DS(test_df,eval_tf)
load=lambda d,sh:DataLoader(d,batch_size=64,shuffle=sh,num_workers=2,pin_memory=False)
tr,va,te=load(train,True),load(val,False),load(test,False)

def make_model(name):
 if name=='ResNet18': m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc=nn.Linear(m.fc.in_features,4); return m
 if name=='MobileNetV3': m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT); m.classifier[3]=nn.Linear(m.classifier[3].in_features,4); return m
 if name=='EfficientNetB0': m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); m.classifier[1]=nn.Linear(m.classifier[1].in_features,4); return m
 if name=='CustomCNN':
  return nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.35),nn.Linear(256,4))

DEVICE=torch.device('cpu'); print('device',DEVICE)
models_to_run=['CustomCNN','ResNet18','MobileNetV3','EfficientNetB0']; allhist={}; summary=[]; reports={}; cms={}
def run_epoch(m,loader,opt=None):
 m.train(opt is not None); lossfn=nn.CrossEntropyLoss(); total=correct=loss_sum=0; ys=[]; ps=[]
 for x,y in loader:
  x,y=x.to(DEVICE),y.to(DEVICE)
  if opt: opt.zero_grad()
  with torch.set_grad_enabled(opt is not None):
   z=m(x); loss=lossfn(z,y)
   if opt: loss.backward(); opt.step()
  pred=z.argmax(1); total+=len(y); correct+=(pred==y).sum().item(); loss_sum+=loss.item()*len(y); ys+=y.tolist(); ps+=pred.tolist()
 return loss_sum/total,correct/total,ys,ps
for name in models_to_run:
 print('TRAIN',name); m=make_model(name).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-4 if name!='CustomCNN' else 3e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
 h={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}; best=(-1,None)
 for ep in range(30):
  tl,ta,_,_=run_epoch(m,tr,opt); vl,va_,_,_=run_epoch(m,va); sched.step();
  for k,v in [('train_loss',tl),('val_loss',vl),('train_acc',ta),('val_acc',va_)]:h[k].append(v)
  if va_>best[0]: best=(va_,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
  print(name,ep+1,round(ta,4),round(va_,4),flush=True)
 m.load_state_dict(best[1]); tl,ta,_,_=run_epoch(m,tr); vl,va_,_,_=run_epoch(m,va); testloss,testacc,ys,ps=run_epoch(m,te)
 rep=classification_report(ys,ps,target_names=classes,output_dict=True,zero_division=0); cm=confusion_matrix(ys,ps)
 reports[name]=rep; cms[name]=cm.tolist(); allhist[name]=h
 summary.append({'model':name,'test_accuracy':testacc,'macro_precision':rep['macro avg']['precision'],'macro_recall':rep['macro avg']['recall'],'macro_f1':rep['macro avg']['f1-score'],'test_loss':testloss,'best_val_accuracy':best[0]})
 torch.save({'model':m.state_dict(),'name':name},OUT/f'{name}.pt')
 pd.DataFrame(rep).T.to_csv(OUT/f'{name}_classification_report.csv'); np.savetxt(OUT/f'{name}_confusion_matrix.csv',cm,fmt='%d',delimiter=',')
# save metrics
pd.DataFrame(summary).to_csv(OUT/'model_summary.csv',index=False); json.dump({'classes':classes,'histories':allhist,'reports':reports,'confusion_matrices':cms},open(OUT/'results.json','w'),indent=2)
# accuracy/loss graphs
sns.set_theme(style='whitegrid'); fig,ax=plt.subplots(2,2,figsize=(13,9));
for name,h in allhist.items():
 ax[0,0].plot(h['train_acc'],label=name); ax[0,1].plot(h['val_acc'],label=name); ax[1,0].plot(h['train_loss'],label=name); ax[1,1].plot(h['val_loss'],label=name)
for a,t in zip(ax.flat,['Training accuracy','Validation accuracy','Training loss','Validation loss']): a.set_title(t); a.set_xlabel('Epoch'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'accuracy_loss_curves.png',dpi=220); plt.close(fig)
# confusion matrix panel
fig,axs=plt.subplots(2,2,figsize=(12,10));
for a,(name,cm) in zip(axs.flat,cms.items()): sns.heatmap(np.array(cm),annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes,ax=a); a.set_title(name); a.set_xlabel('Predicted'); a.set_ylabel('True')
fig.tight_layout(); fig.savefig(OUT/'confusion_matrices.png',dpi=220); plt.close(fig)
# XAI for best model: Grad-CAM, Grad-CAM++, Integrated Gradients on one correctly-classified sample per class
bestname=max(summary,key=lambda x:x['test_accuracy'])['model']; bm=make_model(bestname).to(DEVICE); bm.load_state_dict(torch.load(OUT/f'{bestname}.pt',map_location=DEVICE)['model']); bm.eval()
# locate target convolution layer
def last_conv(m):
 layers=[x for x in m.modules() if isinstance(x,nn.Conv2d)]; return layers[-1]
target=last_conv(bm); acts={}; grads={}
def fwd(mod,inp,out): acts['v']=out
def bwd(mod,gin,gout): grads['v']=gout[0]
h1=target.register_forward_hook(fwd); h2=target.register_full_backward_hook(bwd)
# correctly classified samples
imgs=[]; labels=[]; paths=[]
for idx in range(len(test_df)):
 x,y=test[idx]; withx=x.unsqueeze(0)
 with torch.no_grad(): pred=bm(withx).argmax(1).item()
 if pred==y and y not in labels: imgs.append(x); labels.append(y); paths.append(test_df.iloc[idx].path)
 if len(labels)==4: break
def normalize(a): a=np.maximum(a,0); return (a-a.min())/(a.max()-a.min()+1e-8)
def cam_for(x,method):
 bm.zero_grad(); z=bm(x.unsqueeze(0)); c=z.argmax(1).item(); z[0,c].backward(retain_graph=True); A=acts['v'].detach()[0]; G=grads['v'].detach()[0]
 if method=='Grad-CAM': w=G.mean((1,2)); cam=(w[:,None,None]*A).sum(0)
 elif method=='Grad-CAM++':
  # positive-gradient approximation of Grad-CAM++ weights
  g2=G**2; g3=G**3; den=2*g2+(A*g3).sum((1,2))[:,None,None]+1e-8; alpha=g2/den; w=(alpha*torch.relu(G)).sum((1,2)); cam=(w[:,None,None]*A).sum(0)
 else:
  # integrated-gradient-like input gradient attribution (zero baseline, 20 steps)
  base=torch.zeros_like(x); total=torch.zeros_like(x)
  for a in torch.linspace(.05,1,20):
   xx=(base+a*x).unsqueeze(0).requires_grad_(True); zz=bm(xx)[0,c]; bm.zero_grad(); zz.backward(); total+=xx.grad.detach()[0]
  cam=(total*x).abs().sum(0).detach()
 return normalize(torch.nn.functional.interpolate(cam[None,None],size=(SIZE,SIZE),mode='bilinear',align_corners=False)[0,0].numpy()),c
fig,axs=plt.subplots(4,4,figsize=(12,12));
for r,(x,y) in enumerate(zip(imgs,labels)):
 raw=np.array(Image.open(paths[r]).convert('RGB').resize((SIZE,SIZE)))/255.; axs[r,0].imshow(raw); axs[r,0].set_title(f'Input: {classes[y]}');
 for j,method in enumerate(['Grad-CAM','Grad-CAM++','Integrated gradients'],1):
  heat,p=cam_for(x,method); axs[r,j].imshow(raw); axs[r,j].imshow(heat,cmap='jet',alpha=.45); axs[r,j].set_title(method+'\n'+classes[p])
 for a in axs[r]: a.axis('off')
fig.tight_layout(); fig.savefig(OUT/'xai_comparison.png',dpi=220); plt.close(fig); h1.remove();h2.remove()
# run metadata
meta={'dataset_url':'https://www.kaggle.com/datasets/allandclive/chicken-disease-1/data','seed':SEED,'input_size':SIZE,'epochs':30,'batch_size':64,'optimizer':'AdamW','augmentation':'RandomHorizontalFlip + RandomRotation(20) + ColorJitter(.2,.2,.2)','split':{'train':len(train_df),'validation':len(val_df),'test':len(test_df)},'unique_images_before_augmentation':len(train_df),'unique_images_after_augmentation':len(train_df),'augmented_exposures_over_30_epochs':len(train_df)*30,'best_model':bestname}
json.dump(meta,open(OUT/'metadata.json','w'),indent=2)
print('DONE',bestname,pd.DataFrame(summary))
