from pathlib import Path
import pandas as pd, json, re
root=Path('/home/ubuntu/chicken_study'); p=root/'chicken_disease_classification_article.md'; t=p.read_text()
# Replace the dataset distribution/augmentation paragraph block, removing the pie chart.
old=re.search(r'The observed class counts were.*?The resulting test supports were Coccidiosis 372, Healthy 361, New Castle Disease 84, and Salmonella 394\.',t,re.S).group(0)
new='''The observed class counts were Salmonella 2,625, Coccidiosis 2,476, Healthy 2,404, and New Castle Disease 562. The class distribution is therefore imbalanced, especially for New Castle Disease. The complete source-dataset distribution and the stratified split are reported in Table 1. Because the archive did not provide independent train, validation, and test directories, the CSV rows were split once using stratified sampling with random seed 42.\n\n| Class | Full dataset | Training | Validation | Test | Full-dataset proportion |\n|---|---:|---:|---:|---:|---:|\n| Coccidiosis | 2,476 | 1,733 | 371 | 372 | 30.7% |\n| Healthy | 2,404 | 1,683 | 360 | 361 | 29.8% |\n| New Castle Disease | 562 | 393 | 85 | 84 | 7.0% |\n| Salmonella | 2,625 | 1,837 | 394 | 394 | 32.5% |\n| **Total** | **8,067** | **5,646** | **1,210** | **1,211** | **100.0%** |\n\n**Table 1.** Number of images per class before augmentation and after the stratified split. The validation and test sets were not augmented.'''
t=t.replace(old,new)
# replace preprocessing, architectures, training, eval, xai method paragraphs
start=t.index('### 2.2 Preprocessing and augmentation'); end=t.index('### 2.5 Evaluation measures')
methods='''### 2.2 Preprocessing, augmentation, and class-imbalance handling\n\nImages were resized to 128×128 pixels for computationally consistent CPU training, converted to RGB tensors, and normalized using ImageNet channel means and standard deviations. The four original CNN experiments used random horizontal flipping, random rotation up to 20 degrees, and color jitter. The CNN–Transformer used a stronger regularized training transform: horizontal flipping, low-probability vertical flipping, rotation up to 25 degrees, random resized crop with scale 0.80–1.00, color jitter, and random erasing. Validation and test images received resizing and normalization only.\n\nClass imbalance was handled without duplicating minority images into validation or test sets. Stratified splitting preserved the observed proportions in all partitions, and macro-precision, macro-recall, and macro-F1 were reported so that the majority classes could not dominate the interpretation. For the CNN–Transformer, label smoothing of 0.10, dropout in the convolutional and transformer blocks, AdamW weight decay of 1×10−3, gradient clipping at 1.0, and cosine learning-rate decay were added to reduce memorization. No synthetic oversampling or class reweighting was used; therefore, the reported metrics remain tied to the natural class distribution.\n\n### 2.3 Model architectures\n\nFive models were compared. **CustomCNN**, **ResNet18**, **MobileNetV3-Small**, and **EfficientNet-B0** were defined as in the initial benchmark. The fifth model, **CNN–Transformer**, used four convolutional stages that mapped the image to 256-dimensional spatial tokens, followed by learnable positional embeddings, a two-layer eight-head Transformer encoder, mean token pooling, LayerNorm, dropout, and a four-class classification head. This hybrid design combines local texture extraction from convolutions with global token interaction from self-attention.\n\n### 2.4 Hyperparameters and training procedure\n\nThe original four models used batch size 64, AdamW, cosine-annealing learning-rate decay, and 30 epochs. CustomCNN used learning rate 3×10−4 and the pretrained models used 1×10−4 with weight decay 1×10−4. The CNN–Transformer used batch size 64, learning rate 2×10−4, AdamW weight decay 1×10−3, label-smoothed cross-entropy, gradient-norm clipping at 1.0, and 30 epochs. For every model, the checkpoint with the highest validation accuracy was retained and evaluated once on the untouched test set. This checkpointing is an overfitting-control mechanism, but it does not make an image-level test split equivalent to external validation.\n\n'''
t=t[:start]+methods+t[end:]
# Update XAI wording from four to five and best-model selection
# Existing section starts at 2.5, keep it, but fix count sentence later.
t=t.replace('The best-performing checkpoint was selected by test accuracy only after all four models had completed evaluation.', 'The best-performing checkpoint was selected by test accuracy only after all five models had completed evaluation.')
# Build table rows from reports
models=['CustomCNN','ResNet18','MobileNetV3','EfficientNetB0','CNNTransformer']
labels={'CustomCNN':'CustomCNN','ResNet18':'ResNet18','MobileNetV3':'MobileNetV3-Small','EfficientNetB0':'EfficientNet-B0','CNNTransformer':'CNN–Transformer'}
rows={}
for m in models:
    rep=pd.read_csv(root/'outputs'/f'{m}_classification_report.csv',index_col=0)
    rows[m]=rep

def report_table(m):
    rep=rows[m]
    s='| Class | Precision | Recall | F1-score | Support |\n|---|---:|---:|---:|---:|\n'
    for c in ['Coccidiosis','Healthy','New Castle Disease','Salmonella']:
        r=rep.loc[c]; s+=f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |\n"
    for c in ['accuracy','macro avg','weighted avg']:
        r=rep.loc[c]
        if c=='accuracy': s+=f"| **Accuracy** | — | — | **{r['precision']:.3f}** | {int(r['support'])} |\n"
        else:s+=f"| **{c.title()}** | **{r['precision']:.3f}** | **{r['recall']:.3f}** | **{r['f1-score']:.3f}** | {int(r['support'])} |\n"
    return s
# Build result section
S='''## 3. Result Analysis\n\nAll five models were trained for 30 epochs. Classification reports are presented as complete tables rather than report figures. For each model, **A)** gives the accuracy/loss graph, **B)** gives the complete classification report, **C)** gives the confusion matrix, and **D)** interprets the model-specific result.\n\n'''
summary=pd.read_csv(root/'outputs/model_summary.csv').set_index('model')
# add CNN transformer summary
with open(root/'outputs/CNNTransformer_results.json') as f: cr=json.load(f)
summary.loc['CNNTransformer']={'test_accuracy':cr['test_accuracy'],'macro_precision':cr['report']['macro avg']['precision'],'macro_recall':cr['report']['macro avg']['recall'],'macro_f1':cr['report']['macro avg']['f1-score'],'test_loss':float('nan'),'best_val_accuracy':cr['best_val_accuracy']}
for i,m in enumerate(models,1):
    name=labels[m]; r=rows[m]; a=summary.loc[m]
    if m=='CNNTransformer':
        h=cr['CNNTransformer']; finaltrain=h['train_acc'][-1]; finalval=h['val_acc'][-1]; bestep=max(range(30),key=lambda x:h['val_acc'][x])+1
        interp=f"The CNN–Transformer reached {a.test_accuracy:.2%} test accuracy and macro-F1 {a.macro_f1:.3f}. Its final training accuracy was {finaltrain:.2%}, while validation accuracy was {finalval:.2%}; the best validation checkpoint occurred at epoch {bestep}. The regularization strategy substantially reduced the classic pattern in which training accuracy approaches 100% while validation accuracy plateaus. Although this model was more resistant to memorization, it did not exceed ResNet18 on this dataset."
        graph='CNNTransformer_accuracy_loss.png'; cm='CNNTransformer_confusion_matrix.png'
        accdesc=f"The CNN–Transformer training accuracy finished at {finaltrain:.2%} and validation accuracy at {finalval:.2%}; training and validation loss finished at {h['train_loss'][-1]:.3f} and {h['val_loss'][-1]:.3f}."
    else:
        graph=f'{m}_accuracy_loss.png'; cm=f'{m}_confusion_matrix.png'
        interp={'CustomCNN':'The scratch CNN was the weakest aggregate model. Its lower representation capacity and minority-class F1 of 0.791 indicate that transfer learning is useful for this dataset.','ResNet18':'ResNet18 was the strongest aggregate classifier. It achieved the highest test accuracy and macro-F1, with a strongly diagonal confusion matrix and balanced performance across all four classes.','MobileNetV3':'MobileNetV3-Small was competitive and attractive for resource-constrained deployment. It achieved the highest New Castle Disease recall, but its Healthy–Salmonella confusion reduced its overall score relative to ResNet18.','EfficientNetB0':'EfficientNet-B0 was the strongest minority-class competitor. It achieved the highest New Castle Disease precision and F1 and the lowest test loss in the original benchmark, but its final test accuracy was slightly below ResNet18.'}[m]
        # derive loss graph narrative from source values in results.json
        with open(root/'outputs/results.json') as f: ar=json.load(f)
        h=ar['histories'][m]; accdesc=f"Training accuracy finished at {h['train_acc'][-1]:.2%} and validation accuracy at {h['val_acc'][-1]:.2%}; training loss finished at {h['train_loss'][-1]:.3f} and validation loss at {h['val_loss'][-1]:.3f}."
    S+=f'''### 3.{i} {name}-based results\n\n#### A) Accuracy and loss graph\n\n![{name} accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/{graph})\n\n**Figure {3+i}.** {name} training and validation accuracy and loss over 30 epochs. {accdesc}\n\n#### B) Complete classification report\n\n{report_table(m)}\n\n**Table {2+i}.** Complete test classification report for {name}.\n\n#### C) Confusion matrix\n\n![{name} confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/{cm})\n\n**Figure {8+i}.** Test confusion matrix for {name}.\n\n#### D) Model-based result analysis\n\n{interp}\n\n'''
# comparative table only
S+='''### 3.6 Comparative analysis of all five models\n\n| Model | Architecture and regularization | Test accuracy | Macro precision | Macro recall | Macro-F1 | Validation result | Main error/strength | Final assessment |\n|---|---|---:|---:|---:|---:|---|---|---|\n'''
desc={
'CustomCNN':('Scratch CNN; batch normalization and dropout 0.35','Weakest minority-class F1; useful baseline','Not recommended as final model'),
'ResNet18':('ImageNet-pretrained residual CNN; standard augmentation; validation-checkpoint selection','Balanced class performance; 38/1,211 test errors','**Best overall classifier**'),
'MobileNetV3':('ImageNet-pretrained mobile CNN; compact architecture','Highest New Castle Disease recall; Healthy–Salmonella confusion','Best deployment-oriented alternative'),
'EfficientNetB0':('ImageNet-pretrained compound-scaled CNN','Highest New Castle Disease precision/F1 and lowest original test loss','Best minority-class competitor'),
'CNNTransformer':('CNN tokenization + 2-layer 8-head Transformer; strong augmentation, random erasing, dropout, label smoothing, weight decay, gradient clipping','Smallest train–validation gap; lower final test accuracy','Best overfitting-controlled hybrid, but not best accuracy')}
for m in models:
 a=summary.loc[m]; val=f"Best validation accuracy {a.best_val_accuracy:.2%}"; main,strength,final=desc[m]
 S+=f"| {labels[m]} | {main} | **{a.test_accuracy:.2%}** | {a.macro_precision:.3f} | {a.macro_recall:.3f} | **{a.macro_f1:.3f}** | {val} | {strength} | {final} |\n"
S+='''\nThe final selection is **ResNet18** because it achieved the highest held-out test accuracy (96.86%) and macro-F1 (0.964), while maintaining strong performance for every class. EfficientNet-B0 is the closest accuracy competitor and is preferable when New Castle Disease precision and test loss are prioritized. The CNN–Transformer is the best-controlled model with respect to the train–validation gap, but its 93.97% test accuracy and 0.932 macro-F1 do not exceed the pretrained CNN baselines. Therefore, the evidence supports ResNet18 as the best model for this experiment, subject to the study limitations and the need for external validation.\n\n'''
# Add XAI short subsection, retaining existing figure
S+='''### 3.7 XAI comparison using the best-performing model\n\n![XAI comparison for ResNet18](/home/ubuntu/chicken_study/outputs/xai_comparison.png)\n\n**Figure 19.** Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution for one correctly classified example from each class using ResNet18. The heat maps are qualitative inspection tools, not validated clinical explanations.\n\nGrad-CAM and Grad-CAM++ produced coarse activation maps concentrated mainly around the fecal content. Integrated-gradient-style attribution produced finer, more fragmented input-level patterns. No method is declared more faithful because no region masks, expert annotations, or perturbation-based faithfulness tests were available.\n\n'''
# replace result section
t=t[:t.index('## 3. Result Analysis')]+S+t[t.index('## 4. Discussion'):]
# Discussion tweaks: five models and CNN transformer
p.write_text(t)
print('rewrote article',len(t.splitlines()))
