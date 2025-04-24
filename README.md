# 🔧 Fine-Grained Pruning of VGG on CIFAR-10

This project explores the effect of fine-grained pruning on a pre-trained VGG model using the CIFAR-10 dataset. It includes sensitivity analysis, model sparsity computation, and evaluation of accuracy, precision, recall, and specificity before and after pruning.

---

## 📦 Dataset

- **CIFAR-10**: Automatically downloaded using `torchvision.datasets.CIFAR10`.

---

## 🧠 Model

- **Architecture**: Custom VGG (based on classic VGG11 configuration)
- **Input size**: 32×32 RGB images
- **Number of classes**: 10
- **Pretrained weights**: Downloaded from [MIT HanLab](https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth)

---

## ✂️ Pruning Method

- **Type**: Fine-Grained (magnitude-based) pruning
- **Scope**: All convolutional layers
- **Sparsity levels**: 0.4, 0.5, 0.6
- **Re-applied mask during fine-tuning**: ✅ Yes

---

## 📊 Metrics and Results (Before Fine-Tuning)

All evaluation is done in inference mode on the test dataset.

| Sparsity Level     | Accuracy | Precision | Recall (Sensitivity) | Specificity |
|--------------------|----------|-----------|----------------------|-------------|
| **Original (0.004)** | 92.71%  | 0.9272    | 0.9271               | 0.9918      |
| **0.4 (0.401)**      | 92.25%  | 0.9225    | 0.9225               | 0.9913      |
| **0.5 (0.5005)**     | 90.85%  | 0.9091    | 0.9085               | 0.9898      |
| **0.6 (0.5996)**     | 87.25%  | 0.9091    | 0.9085               | 0.9898      |

> 🧠 Even with a large portion of weights pruned, model accuracy remained high, showing robustness of VGG to sparsity.

---

## 🔁 Fine-tuning Results

### Optimizer: **Adam**, 5 epochs (lr=0.01)

| Sparsity | Accuracy After Fine-Tuning |
|----------|----------------------------|
| 0.4      | 81.34%                     |
| 0.5      | 68.54%                     |
| 0.6      | 72.27%                     |

### Optimizer: **SGD** (momentum=0.9), 5 epochs

| Sparsity | Accuracy After Fine-Tuning |
|----------|----------------------------|
| 0.4      | **92.92%**                 |

---

## 📎 Files

- `efficient.ipynb`: Full notebook of model pruning and experiments
- `conf_mat11.pdf`: Confusion matrix visualization and tabular results
- `README.md`: Project summary (this file)

---

## 🧰 Requirements

```bash
pip install torch torchvision torchprofile matplotlib tqdm scikit-learn
