# Chicken Disease Classification Using Deep Learning and Explainable AI

## Abstract

This study evaluates deep-learning and explainable-artificial-intelligence (XAI) methods for four-class chicken-disease classification from fecal images. The experiment uses a public Chicken Disease Image Classification dataset containing 8,067 RGB JPEG images resized to 224×224 pixels and labeled as Coccidiosis, Healthy, New Castle Disease, or Salmonella. A stratified 70/15/15 split produced 5,646 training, 1,210 validation, and 1,211 test images. Four models were trained for 30 epochs: a custom convolutional neural network (CustomCNN), ResNet18, MobileNetV3-Small, and EfficientNet-B0. Training-only augmentation used horizontal flipping, ±20° rotation, and color jitter; no augmented files were written to disk, so the number of unique images remained 8,067 and the number of unique training images remained 5,646, while the training pipeline exposed each training image to 30 stochastic augmented epochs, equivalent to 169,380 training-image presentations before batch-level variation. ResNet18 was the best model, reaching 96.86% test accuracy and macro-F1 of 0.9642. Its class F1 scores were 0.9717 for Coccidiosis, 0.9610 for Healthy, 0.9467 for New Castle Disease, and 0.9773 for Salmonella. Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution were applied to correctly classified examples from the best model. The explanations are useful for visual inspection, but they are not treated as validated clinical evidence because the dataset has no expert region masks or independent external test set. The main contribution is a leakage-controlled, four-model benchmark with per-class reporting and side-by-side XAI inspection, accompanied by an explicit audit of dataset overlap and XAI omissions in recent poultry studies. The results support ResNet18 as the strongest model in this experiment, but not as a field-ready diagnostic system.

**Keywords:** chicken disease; poultry fecal images; deep learning; transfer learning; ResNet18; Grad-CAM; Grad-CAM++; explainable AI; image classification.

## 1. Introduction

Poultry diseases reduce flock productivity, increase treatment costs, and can be difficult to distinguish from fecal appearance alone. Coccidiosis, Newcastle disease, and Salmonella are clinically important conditions, while healthy samples provide the negative class needed for screening. Computer vision offers a possible low-cost pre-screening aid because mobile images can be processed without laboratory equipment. Recent poultry work has shown that transfer learning can classify fecal images with high reported accuracy, but the evidence is uneven: studies differ in dataset provenance, augmentation, class balance, split strategy, label quality, and whether test images are independent by farm or merely held out at image level [2]–[8].

The background for this study is the growing use of convolutional neural networks (CNNs) and transfer learning in agricultural image analysis. CNNs learn hierarchical texture and shape representations, while pretrained backbones can reduce the data and optimization burden for small or moderately sized agricultural datasets. Lightweight architectures are attractive for mobile deployment, and recent work has also explored end-to-end detection-classification pipelines [6], [7]. However, high accuracy alone does not show that a model uses disease-relevant visual evidence. Explainable AI is therefore important because it can expose whether predictions are concentrated on fecal regions or on spurious background, lighting, or acquisition artifacts [9]–[12].

The existing gap is threefold. First, many poultry studies report one proposed model or a narrow comparison, making architecture-dependent conclusions difficult. Second, the same Tanzania-derived poultry image source appears in several recent studies, yet headline accuracies are often compared without a careful audit of image provenance, class imbalance, and possible overlap [2], [3], [6], [7]. Third, formal XAI is largely absent from the directly comparable poultry studies. Feature-map visualization or attention modules are not equivalent to validated per-prediction explanations, and qualitative heat maps without a region-level reference cannot establish explanation correctness [9]–[12].

Based on this gap, the problem addressed here is: **how can a reproducible four-class chicken fecal-image benchmark compare general-purpose and lightweight deep-learning models while also showing, and appropriately qualifying, the evidence supplied by XAI methods?** The problem includes both predictive discrimination and transparency. It also includes methodological accountability: the study must report the exact image counts, split, augmentation policy, training duration, per-class errors, and limitations that affect comparison with earlier work.

The objectives are to (1) characterize the specified Kaggle dataset and create a stratified leakage-controlled split; (2) train and compare four deep-learning models for at least 30 epochs; (3) report accuracy, loss, confusion matrices, and class-wise precision, recall, and F1; (4) compare Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution on the best model; and (5) compare the findings with seven recent poultry or agricultural disease studies using only traceable publisher or DOI records.

The contributions of this study are:

I. **A reproducible four-model benchmark:** the same split, preprocessing, optimizer family, batch size, random seed, and 30-epoch budget are used for CustomCNN, ResNet18, MobileNetV3-Small, and EfficientNet-B0.

II. **An explicit augmentation accounting protocol:** the study distinguishes unique source images, materialized augmented images, and stochastic augmented training exposures instead of reporting an ambiguous “augmented dataset size.”

III. **A class-sensitive evaluation:** the paper reports per-class support and metrics, which is important because New Castle Disease is substantially underrepresented in the source data.

IV. **A comparative XAI panel:** Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution are applied to the same best model and representative correctly classified examples rather than presenting a single unexplained heat map.

V. **A dataset-overlap-aware literature comparison:** recent benchmark numbers are presented with their dataset and validation caveats; no claim is made that the present image-level split is independent of all previously published work using related Tanzania-derived images.

VI. **A conservative interpretation of explainability:** heat maps are treated as evidence about model attention for inspection, not as proof of clinical causality or explanation faithfulness.

## 2. Materials and Methods

### 2.1 Dataset characteristics

The dataset was downloaded from the user-specified Kaggle page, *Chicken Disease Image Classification* [1]. The Kaggle description states that the images are poultry fecal photographs collected in the Arusha and Kilimanjaro regions of Tanzania between September 2020 and February 2021 using mobile phones and the Open Data Kit application. The four labels are Coccidiosis, Healthy, New Castle Disease, and Salmonella. The page reports that images are resized to 224×224 pixels; inspection of the downloaded archive confirmed 8,067 RGB JPEG files in one `Train` directory and a `train_data.csv` file with 8,067 image-label rows. The Kaggle page lists the dataset license as unknown; consequently, the present work does not claim a particular reuse license beyond using the dataset for analysis as supplied.

The observed class counts were Salmonella 2,625, Coccidiosis 2,476, Healthy 2,404, and New Castle Disease 562. The class distribution is therefore imbalanced, especially for New Castle Disease. The complete source-dataset distribution and the stratified split are reported in Table 1. Because the archive did not provide independent train, validation, and test directories, the CSV rows were split once using stratified sampling with random seed 42.

| Class | Full dataset | Training | Validation | Test | Full-dataset proportion |
|---|---:|---:|---:|---:|---:|
| Coccidiosis | 2,476 | 1,733 | 371 | 372 | 30.7% |
| Healthy | 2,404 | 1,683 | 360 | 361 | 29.8% |
| New Castle Disease | 562 | 393 | 85 | 84 | 7.0% |
| Salmonella | 2,625 | 1,837 | 394 | 394 | 32.5% |
| **Total** | **8,067** | **5,646** | **1,210** | **1,211** | **100.0%** |

**Table 1.** Number of images per class before augmentation and after the stratified split. The validation and test sets were not augmented.

### 2.2 Preprocessing, augmentation, and class-imbalance handling

Images were resized to 128×128 pixels for computationally consistent CPU training, converted to RGB tensors, and normalized using ImageNet channel means and standard deviations. The four original CNN experiments used random horizontal flipping, random rotation up to 20 degrees, and color jitter. The CNN–Transformer used a stronger regularized training transform: horizontal flipping, low-probability vertical flipping, rotation up to 25 degrees, random resized crop with scale 0.80–1.00, color jitter, and random erasing. Validation and test images received resizing and normalization only.

Class imbalance was handled without duplicating minority images into validation or test sets. Stratified splitting preserved the observed proportions in all partitions, and macro-precision, macro-recall, and macro-F1 were reported so that the majority classes could not dominate the interpretation. For the CNN–Transformer, label smoothing of 0.10, dropout in the convolutional and transformer blocks, AdamW weight decay of 1×10−3, gradient clipping at 1.0, and cosine learning-rate decay were added to reduce memorization. No synthetic oversampling or class reweighting was used; therefore, the reported metrics remain tied to the natural class distribution.

### 2.3 Model architectures

Five models were compared. **CustomCNN**, **ResNet18**, **MobileNetV3-Small**, and **EfficientNet-B0** were defined as in the initial benchmark. The fifth model, **CNN–Transformer**, used four convolutional stages that mapped the image to 256-dimensional spatial tokens, followed by learnable positional embeddings, a two-layer eight-head Transformer encoder, mean token pooling, LayerNorm, dropout, and a four-class classification head. This hybrid design combines local texture extraction from convolutions with global token interaction from self-attention.

### 2.4 Hyperparameters and training procedure

The original four models used batch size 64, AdamW, cosine-annealing learning-rate decay, and 30 epochs. CustomCNN used learning rate 3×10−4 and the pretrained models used 1×10−4 with weight decay 1×10−4. The CNN–Transformer used batch size 64, learning rate 2×10−4, AdamW weight decay 1×10−3, label-smoothed cross-entropy, gradient-norm clipping at 1.0, and 30 epochs. For every model, the checkpoint with the highest validation accuracy was retained and evaluated once on the untouched test set. This checkpointing is an overfitting-control mechanism, but it does not make an image-level test split equivalent to external validation.

### 2.5 Evaluation measures

Accuracy was calculated as the proportion of correct test predictions. Precision, recall, and F1 were calculated for each class and macro-averaged across the four classes. Confusion matrices were reported in the order Coccidiosis, Healthy, New Castle Disease, and Salmonella. Because class imbalance affects overall accuracy, macro-F1 and the New Castle Disease row were examined alongside accuracy.

### 2.6 XAI procedure

The best-performing checkpoint was selected by test accuracy only after all five models had completed evaluation. Grad-CAM and Grad-CAM++ were calculated from the last convolutional layer. Integrated-gradient-style attribution used a zero baseline and 20 interpolation steps, with the absolute channel-summed attribution displayed as a heat map. One correctly classified example per class was selected for visualization. The three methods were compared qualitatively on the same inputs. No ground-truth fecal-region masks were available, so intersection-over-union, Dice, insertion/deletion, and faithfulness scores were not claimed. This limitation is important because quantitative XAI evaluation requires additional reference information or carefully designed perturbation tests [9]–[12].

## 3. Result Analysis

All five models were trained for 30 epochs. Classification reports are presented as complete tables rather than report figures. For each model, **A)** gives the accuracy/loss graph, **B)** gives the complete classification report, **C)** gives the confusion matrix, and **D)** interprets the model-specific result.

### 3.1 CustomCNN-based results

#### A) Accuracy and loss graph

![CustomCNN accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/CustomCNN_accuracy_loss.png)

**Figure 4.** CustomCNN training and validation accuracy and loss over 30 epochs. Training accuracy finished at 90.01% and validation accuracy at 92.15%; training loss finished at 0.297 and validation loss at 0.341.

#### B) Complete classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Coccidiosis | 0.963 | 0.914 | 0.938 | 372 |
| Healthy | 0.873 | 0.934 | 0.902 | 361 |
| New Castle Disease | 0.773 | 0.810 | 0.791 | 84 |
| Salmonella | 0.927 | 0.904 | 0.915 | 394 |
| **Accuracy** | — | — | **0.909** | 1211 |
| **Macro Avg** | **0.884** | **0.890** | **0.887** | 1211 |
| **Weighted Avg** | **0.911** | **0.909** | **0.910** | 1211 |


**Table 2.** Complete test classification report for CustomCNN.

#### C) Confusion matrix

![CustomCNN confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/CustomCNN_confusion_matrix.png)

**Figure 9.** Test confusion matrix for CustomCNN.

#### D) Model-based result analysis

The scratch CNN was the weakest aggregate model. Its lower representation capacity and minority-class F1 of 0.791 indicate that transfer learning is useful for this dataset.

### 3.2 ResNet18-based results

#### A) Accuracy and loss graph

![ResNet18 accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/ResNet18_accuracy_loss.png)

**Figure 5.** ResNet18 training and validation accuracy and loss over 30 epochs. Training accuracy finished at 99.82% and validation accuracy at 96.28%; training loss finished at 0.005 and validation loss at 0.149.

#### B) Complete classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Coccidiosis | 0.976 | 0.968 | 0.972 | 372 |
| Healthy | 0.966 | 0.956 | 0.961 | 361 |
| New Castle Disease | 0.941 | 0.952 | 0.947 | 84 |
| Salmonella | 0.970 | 0.985 | 0.977 | 394 |
| **Accuracy** | — | — | **0.969** | 1211 |
| **Macro Avg** | **0.963** | **0.965** | **0.964** | 1211 |
| **Weighted Avg** | **0.969** | **0.969** | **0.969** | 1211 |


**Table 3.** Complete test classification report for ResNet18.

#### C) Confusion matrix

![ResNet18 confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/ResNet18_confusion_matrix.png)

**Figure 10.** Test confusion matrix for ResNet18.

#### D) Model-based result analysis

ResNet18 was the strongest aggregate classifier. It achieved the highest test accuracy and macro-F1, with a strongly diagonal confusion matrix and balanced performance across all four classes.

### 3.3 MobileNetV3-Small-based results

#### A) Accuracy and loss graph

![MobileNetV3-Small accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/MobileNetV3_accuracy_loss.png)

**Figure 6.** MobileNetV3-Small training and validation accuracy and loss over 30 epochs. Training accuracy finished at 99.10% and validation accuracy at 95.79%; training loss finished at 0.029 and validation loss at 0.151.

#### B) Complete classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Coccidiosis | 0.976 | 0.968 | 0.972 | 372 |
| Healthy | 0.946 | 0.928 | 0.937 | 361 |
| New Castle Disease | 0.920 | 0.964 | 0.942 | 84 |
| Salmonella | 0.955 | 0.970 | 0.962 | 394 |
| **Accuracy** | — | — | **0.956** | 1211 |
| **Macro Avg** | **0.949** | **0.957** | **0.953** | 1211 |
| **Weighted Avg** | **0.956** | **0.956** | **0.956** | 1211 |


**Table 4.** Complete test classification report for MobileNetV3-Small.

#### C) Confusion matrix

![MobileNetV3-Small confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/MobileNetV3_confusion_matrix.png)

**Figure 11.** Test confusion matrix for MobileNetV3-Small.

#### D) Model-based result analysis

MobileNetV3-Small was competitive and attractive for resource-constrained deployment. It achieved the highest New Castle Disease recall, but its Healthy–Salmonella confusion reduced its overall score relative to ResNet18.

### 3.4 EfficientNet-B0-based results

#### A) Accuracy and loss graph

![EfficientNet-B0 accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/EfficientNetB0_accuracy_loss.png)

**Figure 7.** EfficientNet-B0 training and validation accuracy and loss over 30 epochs. Training accuracy finished at 99.47% and validation accuracy at 96.78%; training loss finished at 0.014 and validation loss at 0.141.

#### B) Complete classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Coccidiosis | 0.963 | 0.973 | 0.968 | 372 |
| Healthy | 0.955 | 0.942 | 0.948 | 361 |
| New Castle Disease | 0.943 | 0.976 | 0.959 | 84 |
| Salmonella | 0.964 | 0.959 | 0.962 | 394 |
| **Accuracy** | — | — | **0.960** | 1211 |
| **Macro Avg** | **0.956** | **0.963** | **0.959** | 1211 |
| **Weighted Avg** | **0.960** | **0.960** | **0.960** | 1211 |


**Table 5.** Complete test classification report for EfficientNet-B0.

#### C) Confusion matrix

![EfficientNet-B0 confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/EfficientNetB0_confusion_matrix.png)

**Figure 12.** Test confusion matrix for EfficientNet-B0.

#### D) Model-based result analysis

EfficientNet-B0 was the strongest minority-class competitor. It achieved the highest New Castle Disease precision and F1 and the lowest test loss in the original benchmark, but its final test accuracy was slightly below ResNet18.

### 3.5 CNN–Transformer-based results

#### A) Accuracy and loss graph

![CNN–Transformer accuracy and loss curves](/home/ubuntu/chicken_study/outputs/individual_model_figures/CNNTransformer_accuracy_loss.png)

**Figure 8.** CNN–Transformer training and validation accuracy and loss over 30 epochs. The CNN–Transformer training accuracy finished at 94.35% and validation accuracy at 95.62%; training and validation loss finished at 0.470 and 0.446.

#### B) Complete classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Coccidiosis | 0.973 | 0.952 | 0.962 | 372 |
| Healthy | 0.911 | 0.934 | 0.922 | 361 |
| New Castle Disease | 0.904 | 0.893 | 0.898 | 84 |
| Salmonella | 0.944 | 0.944 | 0.944 | 394 |
| **Accuracy** | — | — | **0.940** | 1211 |
| **Macro Avg** | **0.933** | **0.931** | **0.932** | 1211 |
| **Weighted Avg** | **0.940** | **0.940** | **0.940** | 1211 |


**Table 6.** Complete test classification report for CNN–Transformer.

#### C) Confusion matrix

![CNN–Transformer confusion matrix](/home/ubuntu/chicken_study/outputs/individual_model_figures/CNNTransformer_confusion_matrix.png)

**Figure 13.** Test confusion matrix for CNN–Transformer.

#### D) Model-based result analysis

The CNN–Transformer reached 93.97% test accuracy and macro-F1 0.932. Its final training accuracy was 94.35%, while validation accuracy was 95.62%; the best validation checkpoint occurred at epoch 26. The regularization strategy substantially reduced the classic pattern in which training accuracy approaches 100% while validation accuracy plateaus. Although this model was more resistant to memorization, it did not exceed ResNet18 on this dataset.

### 3.6 Comparative analysis of all five models

| Model | Architecture and regularization | Test accuracy | Macro precision | Macro recall | Macro-F1 | Validation result | Main error/strength | Final assessment |
|---|---|---:|---:|---:|---:|---|---|---|
| CustomCNN | Scratch CNN; batch normalization and dropout 0.35 | **90.92%** | 0.884 | 0.890 | **0.887** | Best validation accuracy 92.15% | Weakest minority-class F1; useful baseline | Not recommended as final model |
| ResNet18 | ImageNet-pretrained residual CNN; standard augmentation; validation-checkpoint selection | **96.86%** | 0.963 | 0.965 | **0.964** | Best validation accuracy 96.94% | Balanced class performance; 38/1,211 test errors | **Best overall classifier** |
| MobileNetV3-Small | ImageNet-pretrained mobile CNN; compact architecture | **95.62%** | 0.949 | 0.957 | **0.953** | Best validation accuracy 96.03% | Highest New Castle Disease recall; Healthy–Salmonella confusion | Best deployment-oriented alternative |
| EfficientNet-B0 | ImageNet-pretrained compound-scaled CNN | **95.95%** | 0.956 | 0.963 | **0.959** | Best validation accuracy 97.11% | Highest New Castle Disease precision/F1 and lowest original test loss | Best minority-class competitor |
| CNN–Transformer | CNN tokenization + 2-layer 8-head Transformer; strong augmentation, random erasing, dropout, label smoothing, weight decay, gradient clipping | **93.97%** | 0.933 | 0.931 | **0.932** | Best validation accuracy 95.62% | Smallest train–validation gap; lower final test accuracy | Best overfitting-controlled hybrid, but not best accuracy |

The final selection is **ResNet18** because it achieved the highest held-out test accuracy (96.86%) and macro-F1 (0.964), while maintaining strong performance for every class. EfficientNet-B0 is the closest accuracy competitor and is preferable when New Castle Disease precision and test loss are prioritized. The CNN–Transformer is the best-controlled model with respect to the train–validation gap, but its 93.97% test accuracy and 0.932 macro-F1 do not exceed the pretrained CNN baselines. Therefore, the evidence supports ResNet18 as the best model for this experiment, subject to the study limitations and the need for external validation.

### 3.7 XAI comparison using the best-performing model

![XAI comparison for ResNet18](/home/ubuntu/chicken_study/outputs/xai_comparison.png)

**Figure 14.** Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution for one correctly classified example from each class using ResNet18. The heat maps are qualitative inspection tools, not validated clinical explanations.

Grad-CAM and Grad-CAM++ produced coarse activation maps concentrated mainly around the fecal content. Integrated-gradient-style attribution produced finer, more fragmented input-level patterns. No method is declared more faithful because no region masks, expert annotations, or perturbation-based faithfulness tests were available.

## 4. Discussion

The main measured finding is that ImageNet-pretrained CNNs substantially outperformed the scratch CNN under the same 30-epoch budget, while the regularized CNN–Transformer reduced memorization but did not surpass the CNN baselines. ResNet18 reached 96.86% test accuracy and 0.964 macro-F1, while CustomCNN reached 90.92% and 0.887. Transfer learning likely helped because the dataset has fewer than 600 New Castle Disease images and because pretrained low-level filters provide useful edge and texture representations before task-specific adaptation. EfficientNet-B0 had the strongest New Castle Disease F1 and lowest original test loss, whereas ResNet18 achieved the best overall test accuracy and macro-F1. The CNN–Transformer finished with training accuracy below validation accuracy, showing that the added regularization controlled the train–validation divergence, but its test accuracy was 93.97%.

The New Castle Disease result deserves special attention. The source class had only 562 images, compared with 2,404–2,625 images in the other classes. Stratification preserved the imbalance rather than hiding it through synthetic oversampling. This makes the reported macro metrics more informative than accuracy alone, but the confidence interval around New Castle Disease performance would still be wider than for the larger classes. A study intended for deployment should collect more New Castle Disease cases and include an unknown or unidentifiable class, because every image in a closed four-class classifier must be assigned to one of the four labels.

The results are directionally consistent with the closest directly comparable literature, but they are not interchangeable. Machuve et al. reported up to 98.24% accuracy for Xception with frozen batch normalization on a related Tanzania-derived farm-labeled dataset [2]. Srivastava and Pandey reported 93.23% test accuracy for a custom CNN on an augmented four-class set [3]. Li et al. reported approximately 90% overall accuracy in a mobile proof-of-concept and specifically identified Newcastle disease as a weaker class [5]. Dhungana et al. reported 99.12% for EfficientNet-B0 in a two-stage detection-classification system [6]. These differences may arise from architecture, image resolution, augmentation, train/test construction, model selection, and data overlap. They should not be read as a universal ranking of models.

### Comparison with seven recent studies

| Study | Data and task | Models / XAI | Reported result | Relationship to this study |
|---|---|---|---|---|
| Machuve et al., 2022 [2] | 6,812 farm-labeled Tanzania fecal images plus a 1,255-image laboratory-labeled set; four classes | CNN, VGG16, InceptionV3, MobileNetV2, Xception; no formal XAI | Up to 98.24% with Xception and frozen batch normalization | Closest benchmark; reports stronger headline accuracy but has farm-label noise, limited PCR confirmation, and no formal XAI |
| Srivastava and Pandey, 2023 [3] | 6,812 original images expanded to 9,600 balanced images; four classes | Custom CNN, VGG16, ResNet50, InceptionV3; no XAI | CNN test accuracy 93.23% | Same label family and related source; different augmentation and split protocol |
| Liu et al., 2023 [4] | Poultry feces images; accessible record does not verify class count or split | PoultryNet with MobileNetV3, feature fusion, spatial-channel attention; no XAI | 97.77% accuracy | Supports lightweight attention-based design, but the authoritative record does not establish four-class comparability |
| Wati and Roestam, 2023 [5] | PCR-related chicken fecal images; associated Zenodo record has 6,812 four-class images | YOLOv7 and Soft-NMS; no XAI | Abstract reports bounding-box accuracy above 90%; other metrics not available in the authoritative record | Detection/localization rather than the present image-only classification task |
| Li et al., 2023 [6] | 6,812 four-class Tanzania fecal images; augmentation to 18,719 | VGG19, ResNet101V2, InceptionV3, MobileNetV2, DenseNet169, EfficientNetB2; no XAI | Overall accuracy about 90%; Newcastle accuracy 82% | Strong deployment precedent and useful warning about class imbalance and domain shift |
| Dhungana et al., 2025 [7] | 6,812 PCR-verified fecal/manure images; four classes; detection plus classification | YOLO11 detector and EfficientNet-B0/B1; no formal XAI | EfficientNet-B0 classification accuracy 99.12%; total pipeline 25.8 ms/image | Latest close benchmark, but likely source overlap means not independent external evidence for this experiment |
| Das et al., 2026 [8] | Poultry fecal images tested on farm; exact class count and dataset size not reported in accessible record | MobileNetV2, EfficientNetB0, Xception, ResNet50V2 and ensembles; no XAI | Sum ensemble 99%; voting ensemble 98.5% | Supports ensemble evaluation, but exact four-class comparability is not verified |

The distinctive result of the present study is not a claim of state-of-the-art accuracy. Its contribution is a transparent five-model baseline that reports the exact split and augmentation semantics, complete class-wise tables, per-class errors, overfitting-control measures, and a side-by-side XAI inspection. This distinction matters because the recent literature review found that formal XAI was absent or limited to feature maps in the directly comparable poultry studies [2]–[8]. The present heat maps partially address that omission, but they do not solve explanation validation.

The XAI results also illustrate why accuracy and explanation should be reported separately. All three attribution methods produced plausible attention patterns for selected correct predictions, but they can differ in localization and smoothness. A model can be accurate while relying on acquisition-specific color or background patterns. Recent XAI work in agricultural disease recognition has shown that high classification accuracy does not necessarily imply reliable feature selection and has proposed quantitative evaluation of explanation regions and overfitting to insignificant features [9]. That lesson applies here: without region masks or a perturbation protocol, the present explanations remain diagnostic aids for model inspection rather than evidence of causal reasoning.

### Limitations of the study

The principal limitation is the image-level split. The archive provides no farm identifier, bird identifier, collection-session identifier, or independent external site split. Visually similar images from the same acquisition process could therefore appear across partitions. Because related studies use the same or closely related Tanzania-derived source, the present test set should not be described as external validation against those studies [2]–[7].

The labels and clinical target are also limited. The Kaggle page does not provide complete laboratory-confirmation metadata for every image, and the license is listed as unknown. Fecal appearance can be affected by diet, hydration, age, housing, lighting, and co-infection. The classifier has no clinical history, laboratory result, temporal information, or rejection class.

The experiment was CPU-constrained and used 128×128 input tensors even though the source images are 224×224. The models were trained for 30 epochs with one seed, rather than a large hyperparameter sweep or repeated cross-validation. Confidence intervals, calibration, threshold analysis, inference latency, energy use, and mobile deployment were not evaluated.

The XAI comparison is qualitative and uses one correctly classified example per class. No expert-annotated disease-region masks were available, and no human agreement study, perturbation faithfulness test, or robustness analysis across random seeds was performed. The heat maps should therefore not be used to justify treatment decisions or to claim that the model has identified a pathogen-specific lesion.

## 5. Conclusion and Recommendations

This reproducible experiment compared CustomCNN, ResNet18, MobileNetV3-Small, EfficientNet-B0, and a regularized CNN–Transformer on the specified four-class chicken fecal-image dataset. Under a stratified 70/15/15 split and 30 epochs, ResNet18 produced the best test accuracy at 96.86% and the best macro-F1 at 0.9642. EfficientNet-B0 was the strongest minority-class competitor, MobileNetV3-Small was the strongest compact alternative, CustomCNN was the weakest aggregate model, and the CNN–Transformer showed the smallest train–validation gap after regularization.

The result supports ResNet18 as a strong research baseline for this dataset, not as a validated field diagnostic. Recent studies report higher headline accuracies, but several use related or overlapping Tanzania-derived images and different protocols. Therefore, model claims should be compared only after matching data provenance, split strategy, class balance, augmentation, and validation independence.

Future work should (1) collect an external multi-region test set with farm and acquisition metadata; (2) increase New Castle Disease and difficult Healthy/Salmonella cases; (3) add an unknown/reject class; (4) confirm labels with laboratory testing where feasible; (5) repeat training over multiple seeds and report confidence intervals; (6) evaluate calibration and clinically meaningful operating thresholds; (7) compare explanation faithfulness with deletion/insertion tests and expert-labeled fecal masks; and (8) benchmark quantized ResNet18 or MobileNetV3 on a real mobile device. Any deployment should be positioned as a screening or decision-support tool that complements veterinary and laboratory diagnosis.

## References

[1] Allan D. Clive, “Chicken Disease Image Classification,” Kaggle, dataset, accessed Nov. 9, 2026. [Online]. Available: https://www.kaggle.com/datasets/allandclive/chicken-disease-1/data

[2] D. Machuve, E. Nwankwo, N. Mduma, and J. Mbelwa, “Poultry diseases diagnostics models using deep learning,” *Frontiers in Artificial Intelligence*, vol. 5, Art. no. 733345, 2022, doi: 10.3389/frai.2022.733345.

[3] K. Srivastava and P. Pandey, “Deep learning based classification of poultry disease,” *International Journal of Automation and Smart Technology*, vol. 13, no. 1, Art. no. 2439, 2023, doi: 10.5875/ausmt.v13i1.2439.

[4] X. Liu, Y. Zhou, and Y. Liu, “Poultry disease identification based on light weight deep neural networks,” in *Proc. IEEE 3rd Int. Conf. Computer Communication and Artificial Intelligence*, 2023, pp. 92–96, doi: 10.1109/CCAI57533.2023.10201323.

[5] D. F. Wati and R. Roestam, “Poultry disease detection in chicken fecal images through annotated polymerase chain reaction dataset using YOLOv7 and Soft-NMS algorithm,” in *Proc. 8th Int. Conf. Informatics and Computing*, 2023, pp. 1–7, doi: 10.1109/ICIC60109.2023.10382034.

[6] G. Li, R. S. Gates, and B. C. Ramirez, “An on-site feces image classifier system for chicken health assessment: A proof of concept,” *Applied Engineering in Agriculture*, vol. 39, no. 4, pp. 417–426, 2023, doi: 10.13031/aea.15607.

[7] A. Dhungana, X. Yang, B. Paneru, S. Dahal, G. Lu, and L. Chai, “An integrated deep learning approach for poultry disease detection and classification based on analysis of chicken manure images,” *AgriEngineering*, vol. 7, no. 9, Art. no. 278, 2025, doi: 10.3390/agriengineering7090278.

[8] H. S. Das, K. Talukdar, K. Bora, and K. Borah, “Ensemble-based deep learning approach for early detection of poultry diseases using faecal images,” *British Poultry Science*, vol. 67, no. 4, pp. 676–685, 2026, doi: 10.1080/00071668.2025.2588246.

[9] H. K. Kondaveeti and C. G. Simhadri, “Evaluation of deep learning models using explainable AI with qualitative and quantitative analysis for rice leaf disease detection,” *Scientific Reports*, vol. 15, Art. no. 31850, 2025, doi: 10.1038/s41598-025-14306-3.

[10] A. B. Arrieta *et al*., “Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” *Information Fusion*, vol. 58, pp. 82–115, 2020, doi: 10.1016/j.inffus.2019.12.012.

[11] W. Samek, G. Montavon, S. Lapuschkin, C. Anders, and K.-R. Müller, “Explaining deep neural networks and beyond: A review of methods and applications,” *Proceedings of the IEEE*, vol. 109, no. 3, pp. 247–278, 2021, doi: 10.1109/JPROC.2021.3060483.

[12] P. Nauta, M. van Bree, M. van Gerven, and J. S. K. van der Waa, “From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI,” *Information Fusion*, vol. 91, pp. 464–488, 2023, doi: 10.1016/j.inffus.2022.11.012.

[13] D. P. Blake *et al*., “Re-calculating the cost of coccidiosis in chickens,” *Veterinary Research*, vol. 51, Art. no. 115, 2020, doi: 10.1186/s13567-020-00837-2.

[14] R. Gast *et al*., “Research note: Horizontal transmission and internal organ colonization by Salmonella Enteritidis and Salmonella Kentucky in experimentally infected laying hens in indoor cage-free housing,” *Poultry Science*, vol. 99, no. 12, pp. 6071–6074, 2020, doi: 10.1016/j.psj.2020.08.069.

[15] J. Nezworski *et al*., “A retrospective study of early vs. late virus detection and depopulation on egg laying chicken farms infected with highly pathogenic avian influenza virus during the 2015 H5N2 outbreak in the United States,” *Avian Diseases*, vol. 65, no. 4, pp. 474–482, 2021, doi: 10.1637/aviandiseases-D-21-00038.

[16] J. Wen *et al*., “A rapid novel visualized loop-mediated isothermal amplification method for Salmonella detection targeting at fimW gene,” *Poultry Science*, vol. 99, no. 7, pp. 3637–3642, 2020, doi: 10.1016/j.psj.2020.03.060.

[17] C. Mesa-Pineda, J. L. Navarro-Ruiz, S. López-Osorio, J. J. Chaparro-Gutiérrez, and L. M. Gómez-Osorio, “Chicken coccidiosis: From the parasite lifecycle to control of the disease,” *Frontiers in Veterinary Science*, vol. 8, Art. no. 787653, 2021, doi: 10.3389/fvets.2021.787653.

[18] Y. Lee and H. S. Lillehoj, “Development of a new immunodiagnostic tool for poultry coccidiosis using an antigen-capture sandwich assay based on monoclonal antibodies detecting an immunodominant antigen of *Eimeria*,” *Poultry Science*, vol. 102, no. 5, Art. no. 102790, 2023, doi: 10.1016/j.psj.2023.102790.

[19] C. Janiesch, P. Zschech, and K. Heinrich, “Machine learning and deep learning,” *Electronic Markets*, vol. 31, pp. 685–695, 2021, doi: 10.1007/s12525-021-00475-2.

[20] M. Hussain, “YOLOv1 to v8: Unveiling each variant—A comprehensive review of YOLO,” *IEEE Access*, vol. 12, pp. 42816–42833, 2024, doi: 10.1109/ACCESS.2024.3377633.

[21] A. Dosovitskiy *et al*., “An image is worth 16×16 words: Transformers for image recognition at scale,” in *Proc. Int. Conf. Learning Representations*, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

All numerical results in this article come from the present experiment or from the seven fully verified studies [2]–[8]. References [9]–[21] provide the explainability, agricultural-health, and deep-learning context; none is used to claim an unverified poultry benchmark.

## Reproducibility artifacts

- [Training and validation curves](/home/ubuntu/chicken_study/outputs/accuracy_loss_curves.png)
- [Confusion matrices](/home/ubuntu/chicken_study/outputs/confusion_matrices.png)
- [XAI comparison](/home/ubuntu/chicken_study/outputs/xai_comparison.png)
- [Model summary CSV](/home/ubuntu/chicken_study/outputs/model_summary.csv)
- [Full results JSON](/home/ubuntu/chicken_study/outputs/results.json)
- [Experiment metadata JSON](/home/ubuntu/chicken_study/outputs/metadata.json)
- [Training script](/home/ubuntu/chicken_study/run_study.py)
