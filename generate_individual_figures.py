from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

ROOT=Path('/home/ubuntu/chicken_study'); OUT=ROOT/'outputs'; FIG=OUT/'individual_model_figures'; FIG.mkdir(exist_ok=True)
classes=['Coccidiosis','Healthy','New Castle Disease','Salmonella']
with open(OUT/'results.json') as f: res=json.load(f)
summary=pd.read_csv(OUT/'model_summary.csv')
sns.set_theme(style='whitegrid',context='talk')
for name,h in res['histories'].items():
    # Accuracy and loss figure
    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    epochs=np.arange(1,len(h['train_acc'])+1)
    ax[0].plot(epochs,h['train_acc'],label='Training accuracy',lw=2.5)
    ax[0].plot(epochs,h['val_acc'],label='Validation accuracy',lw=2.5)
    ax[0].set_title(f'{name}: accuracy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].set_ylim(0.6,1.02); ax[0].legend()
    ax[1].plot(epochs,h['train_loss'],label='Training loss',lw=2.5)
    ax[1].plot(epochs,h['val_loss'],label='Validation loss',lw=2.5)
    ax[1].set_title(f'{name}: loss'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Cross-entropy loss'); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG/f'{name}_accuracy_loss.png',dpi=220); plt.close(fig)
    # confusion matrix
    cm=np.array(res['confusion_matrices'][name])
    fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes,ax=ax)
    ax.set_title(f'{name}: confusion matrix'); ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); fig.tight_layout(); fig.savefig(FIG/f'{name}_confusion_matrix.png',dpi=220); plt.close(fig)
    # classwise bar chart
    fig,ax=plt.subplots(figsize=(9,5)); rep.plot(kind='bar',ax=ax); ax.set_ylim(0.7,1.02); ax.set_title(f'{name}: class-wise precision, recall, and F1'); ax.set_ylabel('Score'); ax.set_xlabel(''); ax.legend(loc='lower left',ncol=3); fig.tight_layout(); fig.savefig(FIG/f'{name}_classwise_metrics.png',dpi=220); plt.close(fig)
print('Generated',len(list(FIG.glob('*.png'))),'individual figures in',FIG)
