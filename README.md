# CIFAR10-VehicelAI
For the personal project (homework) on the lecture "Vehicle AI"

본 프로젝트는 CIFAR-10 이미지 분류를 위해 여러 딥러닝 모델을 구현하고 성능을 비교한 프로젝트이다.  
비교 모델은 MLP, Custom CNN, ResNet18, ResNet50, ResNet50 Transfer Learning, ViT Transfer Learning으로 구성하였다.

---

## 1. Project Structure

```text
CIFAR10_project/
├── data/
│   └── cifar-10-batches-py/
│
├── main/
│   ├── main_test.py
│   ├── test_MLP.py
│   ├── test_CNN.py
│   ├── test_ResNet18.py
│   ├── test_ResNet50.py
│   ├── test_ResNet50Transfer.py
│   └── test_ViT.py
│
├── saved_models/
│   ├── BasicNN_MLP_best.pth
│   ├── CustomCNN_best.pth
│   ├── ResNet18_best.pth
│   ├── ResNet50_best.pth
│   ├── ResNet50_Transfer_best.pth
│   └── ViT_B16_Transfer_best.pth
│
├── result_image/
│   └── test 결과 이미지 저장 폴더
│
├── training_result_image/
│   └── 학습 과정 그래프 저장 폴더
│
└── README.md
```

---

## 2. Dataset

본 프로젝트에서는 CIFAR-10 데이터셋을 사용하였다.

CIFAR-10은 총 10개의 클래스로 구성된 이미지 분류 데이터셋이다.

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

데이터셋 구성은 다음과 같다.

| Dataset | Images |
|---|---:|
| Train | 50,000 |
| Test | 10,000 |

`data/` 폴더는 CIFAR-10 데이터셋을 저장하기 위한 폴더이다.  
테스트 코드 실행 시 아래 경로에 데이터가 있어야 한다.

```text
CIFAR10_project/data/cifar-10-batches-py/
```

---

## 3. Models

본 프로젝트에서 비교한 모델은 총 6개이다.

| Model | Description                                   |
|---|-----------------------------------------------|
| BasicNN_MLP | 이미지를 1차원 벡터로 펼친 뒤 Fully Connected Layer로 분류   |
| Custom CNN | Convolution Layer를 사용하여 이미지의 지역적 특징을 추출       |
| ResNet18 | Skip Connection을 적용한 18-layer CNN 구조          |
| ResNet50 | Bottleneck Block을 사용한 더 깊은 Residual Network   |
| ResNet50 Transfer Learning | ImageNet 사전학습 ResNet50을 CIFAR-10에 Fine-tuning |
| ViT Transfer Learning | ImageNet 사전학습 ViT-B/16을 CIFAR-10에 Fine-tuning |

---

## 4. Model Performance

각 모델의 CIFAR-10 test accuracy는 다음과 같다.

| Model | Test Accuracy |
|---|---:|
| MLP | 61.86 % |
| Custom CNN | 91.39 % |
| ResNet18 | 95.11 % |
| ResNet50 | 95.23 % |
| ResNet50 Transfer Learning | 97.67 % |
| ViT Transfer Learning | 98.56 % |

ViT Transfer Learning 모델이 비교군 중 가장 높은 성능을 보였다.

---

## 5. Training Configuration Summary

### MLP

| Configuration | Value |
|---|---|
| Input | 32×32×3 flattened vector |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Output | 10 classes |

MLP는 이미지를 1차원 벡터로 변환하여 입력하기 때문에 픽셀 간의 공간적 위치 관계를 직접 활용하지 못한다.

---

### Custom CNN

| Configuration | Value |
|---|---|
| Input | 32×32×3 |
| Conv Blocks | 3 blocks |
| Pooling | MaxPool2d |
| Regularization | Dropout |
| Loss | CrossEntropyLoss |
| Output | 10 classes |

Custom CNN은 convolution 연산을 통해 이미지의 지역적 특징을 추출한다.  
MLP보다 이미지의 공간 정보를 잘 활용할 수 있어 성능이 크게 향상되었다.

---

### ResNet18 / ResNet50

| Configuration | ResNet18 | ResNet50 |
|---|---|---|
| Input | 32×32×3 | 32×32×3 |
| Main Block | Basic Block | Bottleneck Block |
| Residual Connection | O | O |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Output | 10 classes | 10 classes |

ResNet은 Skip Connection을 통해 입력 정보를 다음 layer로 직접 전달한다.  
이를 통해 깊은 네트워크에서도 gradient vanishing 문제를 완화하고 안정적인 학습이 가능하다.

---

### ResNet50 Transfer Learning

| Configuration | Value |
|---|---|
| Pretrained Weight | ImageNet pretrained ResNet50 |
| Input Size | 224×224 |
| Normalization | ImageNet mean/std |
| Classifier | FC layer 1000 → 10 |
| Loss | CrossEntropyLoss |
| Output | 10 classes |

ResNet50 Transfer Learning은 ImageNet으로 사전학습된 가중치를 초기값으로 사용한다.  
기존 ResNet50이 학습한 일반적인 이미지 특징을 활용하기 때문에 CIFAR-10에서도 높은 성능을 보였다.

---

### ViT Transfer Learning

| Configuration | Value |
|---|---|
| Pretrained Weight | ImageNet pretrained ViT-B/16 |
| Input Size | 224×224 |
| Patch Size | 16×16 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Normalization | ImageNet mean/std |
| Output | 10 classes |

ViT는 이미지를 여러 개의 patch로 나누고, 각 patch 간의 관계를 Self-Attention으로 학습한다.  
CNN이 지역적 특징을 중심으로 학습하는 것과 달리 ViT는 이미지 전체의 전역적 관계를 반영할 수 있다.

---

## 6. Test Code

각 모델은 개별 테스트 코드로 실행할 수 있다.

```bash
python main/test_MLP.py
python main/test_CNN.py
python main/test_ResNet18.py
python main/test_ResNet50.py
python main/test_ResNet50Transfer.py
python main/test_ViT.py
```

각 테스트 코드는 다음 작업을 수행한다.

```text
1. saved_models/ 폴더에서 학습된 .pth 파일 로드
2. CIFAR-10 test set 10,000장 전체 평가
3. Test Loss 및 Test Accuracy 출력
4. Confusion Matrix 출력
5. Sample Prediction 이미지 출력
```

테스트 과정에서는 모델 학습이 이루어지지 않는다.  
즉, weight update 없이 저장된 모델의 성능만 평가한다.

---

## 7. Main Test UI

`main_test.py`를 실행하면 UI 창에서 테스트할 모델을 선택할 수 있다.

```bash
python main/main_test.py
```

선택 가능한 모델은 다음과 같다.

```text
1. BasicNN_MLP
2. CustomCNN
3. ResNet18
4. ResNet50
5. ResNet50 Transfer
6. ViT-B/16 Transfer
```

UI에서 모델을 선택하고 `Run Test` 버튼을 누르면 해당 모델의 test accuracy와 confusion matrix가 출력된다.

`main_test.py`는 결과 이미지를 저장하지 않고 화면에만 표시한다.

---

## 8. Confusion Matrix

Confusion Matrix는 실제 클래스와 모델이 예측한 클래스를 비교하는 표이다.

```text
행: 실제 정답 클래스
열: 모델이 예측한 클래스
대각선 값: 올바르게 분류한 개수
비대각선 값: 잘못 분류한 개수
```

대각선 값이 클수록 모델이 해당 클래스를 잘 분류한 것이다.  
비대각선 값이 큰 경우, 해당 클래스 간 혼동이 많다는 의미이다.

---

## 9. Sample Prediction

Sample Prediction은 test set 중 일부 이미지를 선택하여 모델의 예측 결과를 시각화한 것이다.

```text
GT   : Ground Truth, 실제 정답
Pred : Prediction, 모델 예측 결과
```

`main_test.py`에서는 옵션에 따라 고정된 샘플 또는 랜덤 샘플을 확인할 수 있다.

---

## 10. Requirements

본 프로젝트 실행을 위해 필요한 주요 라이브러리는 다음과 같다.

```text
torch
torchvision
numpy
matplotlib
scikit-learn
tkinter
```

일반적인 설치 명령어는 다음과 같다.

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

`tkinter`는 대부분의 Python 설치 환경에 기본 포함되어 있다.

---

## 11. Notes

- Scratch 모델인 MLP, Custom CNN, ResNet18, ResNet50은 CIFAR-10 기준 normalization을 사용한다.
- Transfer Learning 모델인 ResNet50 Transfer와 ViT Transfer는 ImageNet 기준 normalization을 사용한다.
- Transfer Learning 모델은 테스트 시 pretrained weight를 다시 다운로드하지 않고, `saved_models/`에 저장된 fine-tuned weight를 불러온다.
- Test accuracy는 CIFAR-10 test set 10,000장 전체를 한 번 평가한 결과이다.
- Sample Prediction 이미지는 test 결과 일부를 시각적으로 확인하기 위한 용도이다.

---

## 12. Conclusion

본 프로젝트에서는 CIFAR-10 이미지 분류 문제에 대해 다양한 신경망 구조를 비교하였다.

MLP는 이미지의 공간 정보를 직접 활용하지 못해 가장 낮은 성능을 보였다.  
Custom CNN은 convolution을 통해 지역적 특징을 학습하여 MLP 대비 큰 성능 향상을 보였다.  
ResNet 계열은 residual connection을 통해 깊은 구조에서도 안정적인 학습이 가능했고, 높은 분류 성능을 보였다.  
Transfer Learning을 적용한 ResNet50과 ViT는 사전학습된 이미지 특징을 활용하여 scratch 학습 모델보다 더 높은 성능을 보였다.

최종적으로 ViT Transfer Learning 모델이 비교군 중 가장 높은 성능을 보였다.
