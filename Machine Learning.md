* 좋은 모델의 정의는 무엇일까요?
    * 새로운 데이터에 대해서도 성능이 준수한, 일반화된 모델의 경우 좋은 모델이라고 할 수 있을 것 같습니다.

* 알고 있는 metric에 대해 설명해주세요.
    * 분류(Classification)
        * 정확도(Accuracy): 전체 데이터 중에 정확히 예측한 데이터의 비율
            * Imbalanced Data에는 주의해야함
        * 오차행렬(Confusion Matrix): 모델이 예측을 얼마나 헷갈리고 있는지를 보여주는 지표
        * 정밀도(Precision): 긍정으로 예측한 데이터 중에 실제로 긍정인 데이터의 비율
            * 정밀도와 재현율은 트레이드오프 관계
        * 재현율(Recall): 실제로 긍정인 데이터 중에 긍정으로 예측한 데이터의 비율
            * 정밀도와 재현율은 트레이드오프 관계
        * Fβ-Score: 정밀도와 재현율의 가중조화평균
            *  β가 1인 경우를 F1-Score라고 함
        * ROC(Receiver Operating Characteristic) Curve: FPR(False Positive Rate)가 변할 때 TPR(True Positive Rate)가 어떻게 변하는지를 나타내는 곡선
            * 임계값(Threshold)을 변경함으로써, FPR과 TPR을 계산함
        * AUC(Area Under Curve): ROC 곡선 아래의 면적
            * AUC가 0.5보다 작게 나오는 경우 학습이 잘못 되었는지를 의심해봐야 됨
            * AUC는 척도 불변, 임계값 불변 특성을 가지고 있음
    * 회귀(Regression)
        * MAE(Mean Absolute Error): 예측값과 관측값 차이의 절댓값의 평균
        * MSE(Mean Squared Error): 예측값과 관측값 차이의 제곱의 평균
            * MAE보다 이상치에 민감함
        * RMSE(Root Mean Squared Error): Root가 적용된 MSE
        * 결정계수(R Squared): 분산을 기반으로, 관측값의 분산 대비 예측값의 분산 비율
            * R² = SSE(Explained Sum of Squares) / SST(Total Sum of Squares) = (회귀식에 의해 설명되는 변동) / (전체 변동)

* 결측치(missing value)가 있을 경우 채워야 할까요?
    * 결측치의 비율에 따라 다르게 대응합니다.
        * ~ 10%: Imputation
        * 10% ~ 50%: Prediction by Model
        * 50% ~: Delete
    * XGBoost의 경우, 모델이 결측치에 대응할 수 있습니다.

* 이상치(Outlier)의 판단하는 기준은 무엇인가요?
    * 이상치(Outlier)는 일반적인 패턴을 벗어난 데이터를 의미합니다.
    * IQR(Inter Quantile Range): Q3-Q1을 IQR이라고 하고, 최댓값(Q3+IQR×1.5)보다 크거나 최솟값(Q1-IQR×1.5)보다 작은 경우 이상치로 판단합니다.
    * Z-Score: 가우시안 분포를 따를 때 사용할 수 있으며, 임계값을 벗어나는 경우 이상치로 판단합니다.

* 정보량? 엔트로피(entropy)? 정보이득(Information Gain)?
    * 정보량은 불확실성을 정량화 한 것으로, 발생 확률이 작을수록 큰 값을 가집니다.
    * 엔트로피(entropy)는 정보량의 기댓값입니다.
    * 정보이득(Information Gain)은 감소되는 엔트로피의 양(기존 엔트로피 - 현재 엔트로피)을 의미합니다.

* 스케일링을 해야하는 이유?
    * 피처 간에 스케일이 심하게 차이나는 경우 각 피처의 기울기가 달라져 경사하강법에서 불안정한 학습을 유발할 수 있습니다. 그래서 비슷하거나 동일한 스케일로 변경하는 것이 필요합니다.
    * 정규화(Normalization): 피처를 0과 1사이의 값으로 변환
        * Min-Max Normalization
        * 상한과 하한이 정해져 딥러닝에 유리함
        * 이상치에 영향이 큼
    * 표준화(Standardization) : 피처를 평균 0, 표준편차 1로 변환
        * Z-Score Standardization
        * 이상치에 영향이 적음
        * 피처 간에 동일한 스케일을 가지지 않음(= 상한과 하한이 정해지지 않음)

* 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
    * Newton’s Method: Convex한 2차 테일러 근사식에서 최솟값을 찾으며, 원래 식의 최솟값에 근사적으로 접근하는 방법입니다.
        * 근사식에서 그래디언트가 0인 지점을 찾고, 해당 지점을 기준으로 한 근사식에서 그래디언트가 0인 지점을 찾는 과정을 반복해서 해에 접근함
        * Gradient Descent보다 많은 계산량과 큰 메모리를 요구함
    * Gradient Descent: 헤시안 행렬 대신 1/t로 스칼라배된 정방행렬을 사용한 2차 테일러 근사식에서 최솟값을 찾으며, 원래 식의 최솟값에 근사적으로 접근하는 방법입니다.
        * 근사식에서 그래디언트가 0인 지점을 찾고, 해당 지점을 기준으로 한 근사식에서 그래디언트가 0인 지점을 찾는 과정을 반복해서 해에 접근함

* 경사하강법(Gradient Descent)?
    * 경사하강법(Gradient Descent): 어떤 함수의 최솟값을 찾기 위해 Gradient의 반대 방향으로 이동해 가는 방법입니다.
    * 딥러닝의 경우 단순한 회귀식처럼 단번에 최적의 가중치를 찾는 것은 어렵기 때문에 경사하강법을 사용함
    * 극소 영역에서는 비선형 변환 또한 선형 변환으로 바라볼 수 있기 때문에 딥러닝에 경사하강법을 적용할 수 있음
    * Newton’s Method의 특이 케이스(헤시안 행렬 대신 1/t로 스칼라배된 정방행렬을 사용)

* 경사하강법 중에 때때로 Loss가 증가하는 이유는?
    * Local Minima에서 빠져나오는 경우 Loss가 일시적으로 증가할 수 있습니다.
    * 학습률이 커서 Global Minima 근처를 선회하는 현상에서 발생 할 수 있을 것 같습니다.

* 역전파(BackPropagation)에 대해서 쉽게 설명 한다면?
    * 역전파(BackPropagation): Loss에 대한 Gradient를 출력층부터 계산하여 입력층 방향으로 전파시키는 것입니다.
        * 연쇄법칙(Chain Rule)이 이용됨

* Global Minimum(Minima)와 Local Minimum(Minima)에 대해 설명해주세요.
    * Global Mimimum은 손실함수의 손실이 최소화 되는 지점입니다.
    * Local Minimum은 손실이 최소화 될 수 있는 후보 중에 Global Minumum을 제외한 지점들입니다.
        * 모멘텀과 적응적 학습률, 학습률 스케줄링을 통해 해결할 수 있음

* Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
    * 고차원 공간에서 모든 축이 아래로 볼록 형태가 형성될 확률은 희박함
    * 오히려, Local Minima보다 빈번하게 발생하는 것은 Critical Point인 Saddle Point임
        * Critical Point: Gradient가 0인 지점
        * Saddle Point: 위아래 볼록 형태가 형성되며, Gradient가 0인 지점
        * Local Minima처럼 모멘텀과 적응적 학습률, 학습률 스케줄링을 통해 해결할 수 있음

* 모멘텀(Momentum)?
    * 모멘텀(Momentum): Gradient의 방향을 유지시켜주는 기법입니다.

* 적응적 학습률(Adaptive Learning Rate)?
    * 적응적 학습률(Adaptive Learning Rate): 파라미터마다 자신의 상황에 맞게 학습률을 조정하는 기법입니다.
        * 기울기가 가파를수록 조금만 이동하고, 완만할수록 많이 이동함

* Adam Optimizer?
    * Gradient의 1th Moment와 2th Moment를 추정해서 모멘텀과 적응적 학습률을 최적화에 적용한 알고리즘입니다. 
    * 특히, 좋은 추정량이 될 수 있도록 불편성을 만족하게 보정한 것이 큰 특징입니다.

* 정체기(Plateau)?
    * 정체기(Plateau): Gradient가 0에 가까워져 Loss가 업데이트 되지 않는 현상입니다.

* Warm Restart?
    * Warm Restart: 학습 중간중간에 학습률을 증가시키는 것입니다.
        * 큰 폭의 파라미터 업데이트를 만들어 Local Minima에서 빠져나오도록 함

* 차원의 저주에 대해 설명해주세요.
    * 데이터 차원이 증가할수록 해당 공간의 크기가 기하급수적으로 증가하여 데이터 간 거리가 멀어지고 과적합(Overfitting)을 유발시키는 현상을 의미합니다.
        * 데이터를 늘리거나, 데이터 차원을 줄여서 해결할 수 있음

* 차원 축소(Dimension Reduction)기법으로 보통 어떤 것들이 있나요?
    * Feature Selection: 기존 Feature Set에서 주요 Feature Subset을 선택함
    * Feature Extraction: 기존 Feature Set을 정보손실이 적으면서 저차원 형태로 매핑함

* SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요?
    * SVM은 Margin이라는 개념을 도입해 Decision Boundary를 정의해서 다양한 문제를 해결하는 모델입니다.
    * 커널트릭을 통해 비선형 특성을 추가해서 선형분류가 가능하도록 차원을 확장시킵니다.
        * 커널트릭(Kernel Trick): 실제로는 특성을 추가하지 않으면서 새로운 특성을 추가한 것과 같은 결과를 얻는 스킬

* 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
    * 거대하고 높은 품질의 데이터셋이 축적되어 Inductive Bias가 적은 모델들이 높은 일반화 성능을 내면서 딥러닝의 혁신적인 발전이 시작되었다고 생각합니다.
    * 또한 하드웨어에 발전으로 Large Model을 다룰 수 있게 된 것도 중요한 점이라고 생각합니다.

* K-means의 대표적 의미론적 단점은 무엇인가요?
    * K-Means: K개의 군집 개수를 정하고, 군집의 중심점을 갱신해나가는 클러스터링 알고리즘
    * K를 몇 개로 설정하느냐에 따라 성능이 달라짐
    * 초기 중심점을 어디에 설정하느냐에 따라 성능이 달라짐
    * 노이즈가 많은 경우 성능이 떨어짐

* Training 세트와 Test 세트를 분리하는 이유는?
    * 처음 보는 데이터에도 강건하게 예측을 하는지 판단하기 위해 Test 세트를 따로 만듬
        * Test 세트를 통해 일반화 성능을 추정함
    * Test 세트에 대한 정보를 조금이라도 얻으면 Test 세트가 오염되었다고 함

* Validation 세트가 따로 있는 이유는?
    * Test 세트에 과적합 되는 데이터 스누핑 편향을 방지하기 위해 Validation 세트를 사용함
        * Validation 세트를 통해 성능 평가 및 하이퍼 파라미터를 튜닝함

* Cross Validation은 무엇인가요?
    * Cross Validation: 학습 데이터를 K개의 Fold로 나누어, 그 중 하나를 검증 데이터로 사용해 K번 학습해서, K개의 모델 결과를 평균내어 최종 결과로 사용하는 방법입니다.
        * 여러 개의 검증 데이터를 사용하기 때문에 홀드 아웃 검증에 비해 일반화 된 검증 결과를 얻을 수 있음
        * 전반적으로 데이터가 적으면 검증 데이터셋의 크기도 작아져 신뢰성을 잃을 수 있는데, 이러한 문제를 해결 할 수 있음
        * K번의 학습을 하기 때문에 시간이 오래 걸림

* XGBoost? LGBM? Catboost?
    * XGBoost
        * 병렬처리: 분기마다 처리하거나 분기 내에 계산을 할 때 병렬처리가 일어남
        * Regularization: minimum information gain을 만족하는 경우에만 분기가 일어남
        * Pruning: gamma보다 작은 information gain을 가지고 있는 분기를 제거함
        * 결측치 처리: Training Loss를 감소시키는 최적의 값으로 대치함
    * LGBM
        * GOSS(Gradient-based One Side Sampling)
        * EFB(Exclusive Feature Bundling)
    * Catboost
        * Ordering Principle

* 앙상블(Ensemble) 방법엔 어떤 것들이 있나요?
    * 앙상블은 여러 개의 모델들의 결과를 조합해서 최종 결과를 만들어내는 방법입니다.
        * 모델마다 잘 예측하는 영역이 다르기 때문에 예측의 결과를 조합하면 상호보완적인 결과를 도출해낼 수 있음
        * 개별 모델에서 과적합이 발생할 수 있는데, 예측 결과를 조합하면 과적합이 줄어들어 일반화 성능이 향상됨
    * Voting: 다른 종류의 여러 분류기들의 투표를 통해 최종 예측 결과를 결정하는 방식
    * Bagging: 리샘플링을 통해 같은 종류의 모델을 병렬로 학습하고, 모델들의 결과를 집계해서 최종 예측 결과를 결정하는 방식
    * Boosting: 약한 학습기들을 직렬로 연결하여 앞의 모델을 보완하면서 강한 학습기를 만들어내는 방식
    * Stacking: 여러 모델들이 예측한 결과들을 다시 데이터로 사용해서 최종 예측을 수행하는 방식

* 최소자승법(OLS, Ordinary Least Squres)은 무엇인가요?
    * 예측값과 관측값의 차이 제곱의 합이 최소가 되는 해를 구하는 방법입니다.
    * OLS 식에서 그래디언트가 0인 등식을 통해 정규방정식을 유도할 수 있습니다.

* 머신러닝과 딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?
    * 머신러닝: 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘
        * 데이터 품질에 매우 의존적임
    * 딥러닝: 여러 층을 가진 인공신경망(ANN, Artificial Neural Network)을 사용한 머신러닝
        * 큰 데이터셋과 긴 학습시간이 요구됨
    * 머신러닝은 데이터에서 어떤 특징을 추출해서 학습할지 사람이 직접 분석하고 판단해야하는 반면, 딥러닝은 데이터에서 자동으로 특징을 추출해서 학습할 수 있음

* 머신러닝(machine learning)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
    * 머신러닝적 접근방법: 높은 예측률을 위해 복잡한 모델을 사용함
    * 통계적 접근방법: 해석 가능하도록 단순한 모델을 사용함

* 비용함수(Cost Function)?
    * 비용함수(Cost Function): 데이터에 대한 예측값과 관측값 사이에 오차를 계산해서 현재 예측이 얼마나 올바른지 정량화 해줄 수 있는 함수입니다.
        * Cost Function을 최소화함으로써 모델이 적절한 표현을 갖추도록 학습시킬 수 있음
    * 손실함수(Loss Function): 하나의 데이터에 대해서 오차를 계산하는 함수
    * 비용함수(Cost Function): 전체 데이터에 대해서 오차를 계산하는 함수
    * 목적함수(Objective Function): 값을 최소화하거나 최대화하는 것을 목적으로 하는 함수

* Linearity?
    * Linearity: 선의 형태를 가지는 성질로, 가산성(Additivity)과 동차성(Homogeneity)를 만족해야합니다.
        * 가산성(Additivity): f(x+y) = f(x) + f(y)
        * 동차성(Homogeneity): f(ax) = af(x)
    * Non-Linearity: 선의 형태를 가지지 않는 성질로, 가산성과 동차성이 만족하지 않습니다.

* Activation Function?
    * Activation Function: 비선형성을 제공해 레이어를 깊게 쌓을 수 있어서 모델이 더 복잡한 특징을 추출 할 수 있도록 합니다.
        * 데이터에 대한 모델의 표현력을 향상시킬 수 있음
        * 연구동향: Unbounded Above, Bounded Below, Zero-Centric(Centered)
    * Sigmoid
        * Bounded Above 형태를 가지고 있어서 Gradient Vanishing 문제를 유발함
        * Non Zero-Centric 형태를 가지고 있어서 지그재그 현상과 같이 불안정한 학습을 야기함
        * Exponetial 함수를 포함하고 있기 때문에 연산량이 큼
    * Tanh
        * Bounded Above 형태를 가지고 있어서 Gradient Vanishing 문제를 유발함
        * Exponetial 함수를 포함하고 있기 때문에 연산량이 큼
    * ReLU
        * Unbounded Above 형태를 가지고 있어서 Gradient Vanishing 문제가 없음
            * 활성화된(양수구간) 노드의 경우 기울기가 1임 
        * 간단하고 계산이 효율적임
        * Non Zero-Centric 형태를 가지고 있어서 지그재그 현상과 같이 불안정한 학습을 야기함
        * Dying ReLU 문제를 야기함
            * Dying ReLU = Dead ReLU(= Neurons)
            * 노드의 출력이 0보다 작아서 활성화가 안되며 오차역전파 전달이 안되는 현상
            * 노드에 아주 조금의 편향(Bias, Wx+"b")주어(= ReLU를 왼쪽으로 조금 이동) 활성화 될 가능성을 높이는 스킬이 존재함
                * Batch Normalization을 쓸 경우 해당 스킬은 의미가 없어짐
    * LeakyReLU: Dying ReLU를 해결하기 위해 음수구간에 작은 선형성을 제공함
        * Unbounded Below 형태를 가지고 있어서 음수구간에도 선형성이 생기면서 복잡한 패턴을 학습 못할 수 있음

* Tensorflow, PyTorch 특징과 차이가 뭘까요?
    * Tensorflow: Define and Run
    * Pytorch: Define by Run

* 과적합(Overfitting)일 경우 어떻게 대처해야 할까요?
    * 과적합(Overfitting)
        * 모델 내에 노드와 특정 샘플(특히, 이상치) 간에 의존성이 생기거나 노드 간에 의존성이 생겨서 발생하는 문제
        * 이상치나 데이터에 존재하는 노이즈와 같은 일반화와 관련 없는 요소들의 패턴까지 학습하면서 발생하는 문제
    * 모델 복잡도(Complexity, Capacity)를 줄임
    * Data
        * 데이터 증강(Data Augmentation)
        * 데이터 품질 개선
            * 노이즈(Noise) 감소
            * 이상치(Outlier) 제거
    * Hyper Paramerter
        * 학습률(Learning Rate) 감소 
        * 배치사이즈(Batch Size) 증가
            * 이상치의 영향을 줄임
    * Regularization: 모델 학습에 패널티를 부여해서 과적합 문제를 해결하는 기법
        * Early Stopping
        * L1, L2 Regularization, Weight Decay
        * Dropout
    * Batch Normalization

* 데이터 증강(Data Augmentation)?
    * 데이터에 인위적인 변화를 주어 데이터의 수를 늘리는 방법입니다.

* Hyper Parameter?
    * Hyper Parameter: 사용자가 직접 설정하는 값입니다.
    * 휴리스틱하게 설정하거나 하이퍼 파라미터 튜닝기법을 사용해서 설정합니다.
        * Manual Search, Grid Search, Random Search, Bayesian Optimization
    * 손실함수, 활성화함수 등도 유연하게 하이퍼 파라미터로 이해함

* Early Stopping?
    * Validation Loss가 올라가는 시점에 학습을 종료하는 방법입니다.

* L1, L2 규제(Regularization)?
    * 모델 파라미터가 큰 스케일의 값을 가지면서 과적합이 발생하기 때문에 모델 파라미터가 작은 스케일을 가지도록 패널티를 부여하는 것이 L1, L2 규제의 핵심 아이디어입니다.
    * L1 규제는 손실함수에 L1 Norm을 더하는 것으로, 특정 파라미터가 0이 되는 현상을 가지고 있습니다.
    * L2 규제는 손실함수에 L2 Norm(정확히는, 루트없음)을 더하는 것으로, 전반적으로 파라미터들을 작게 만들어 이상치나 노이즈가 있는 데이터에 적합합니다.

* Weight Decay?
    * 경사하강법을 통해 파라미터를 업데이트 할 때, 기존 파라미터를 일정 비율 감소시키는 방법입니다.

* Dropout?
    * 과적합은 모델 내에 노드와 특정 샘플(특히, 이상치) 간에 의존성이 생기거나 노드 간에 의존성이 생겨서 발생하는 문제로, Dropout은 모델 내에 노드들을 랜덤하게 선택하기 때문에 앞서 말씀드린 의존성을 끊어내는 것을 기대할 수 있기 때문에 과적합 문제에 대응할 수 있습니다. 
    * 또한 노드 간에 의존성을 끊어냈기 때문에 하나의 모델 아키텍처에서 상관관계가 적은 다양한 결과를 Aggregation 하는 것으로 해석할 수 있어, 마치 상관관계가 적은 모델들의 앙상블을 사용하는 것과 같은 효과를 가져와 성능 향상을 기대해볼 수 있습니다.
    * Fully Connected Layer에서는 노드 단위, Convolution Layer에서는 채널 단위로 적용됨
    * 학습 시에만 적용하고, 추론 시에는 적용하지 않음

* Batch Normalization?
    * Normalization 이라는 네이밍과 다르게, mini-batch 단위의 Standardization과 Destandardization으로 이루어져 있습니다. 
    * Standardization는 평균을 빼주고 표준편차로 나눠주는 작업으로 과적합 문제를 해소하고, 수렴속도를 높일 수 있습니다. 
    * Destandardization의 경우 학습가능한 변수들을 통해 모델 스스로 적절한 출력구간으로 Scaling 및 shift를 할 수 있도록 하여, Gradient Vanishing과 Non-liearlity 간에 발생하는 트레이드오프의 적절한 지점을 찾을 수 있습니다.
    * 추론 시에는 Moving Average를 통해 계산된 평균과 분산을 사용함
        * 추론할 때 Batch Size가 달라질 수 있음
        * Batch Normalization이 학습데이터로 학습되었기 때문에 일관성을 보존하기 위해서임

* 가중치 초기화(Weight Initialization) 방법에 대해 말해주세요.
    * 가중치를 잘 초기화하면 기울기 소실문제나 Local Minima 문제를 해결 할 수 있습니다.
    * LeCun Initialization
    * Xavier Initialization: sigmoid, tanh에 주로 사용함
    * He Initialization: ReLU에 주로 사용함

* SGD에서 Stochastic의 의미는?
    * (Mini) Batch를 구성하는 데이터가 학습데이터에서 무작위로 선택된다는 것을 의미함

* CNN이 아닌 MLP로 해도 잘 될까?
    * CNN은 Locality와 Parameter Sharing을 통해 동일한 개체가 이미지 내에 위치가 바뀌어도 똑같은 특징을 추출할 수 있기 때문에 복잡한 패턴을 잘 포착할 수 있어서 일반화가 쉽습니다.
        * 여러 층을 통해 계층적으로 특징을 추출하면서 복잡한 패턴을 포착해낼 수 있음
    * 반면에 MLP는 공간적 관계정보를 활용하지 않기 때문에 동일한 개체가 이미지 내에 위치가 바뀌면 다른 특징을 추출하기 때문에 복잡한 패턴을 잘 포착하지 못해서 일반화가 어렵습니다.

* RNN이 아닌 MLP로 해도 잘 될까?
    * RNN은 Sequentiality(Time Dependencies)와 Parameter Sharing을 통해 등장 시점이 바뀐 동일한 순서에 대해 똑같은 특징을 추출할 수 있어서 일반화가 쉽습니다.
    * 반면에 MLP는 독립적으로 처리하기 때문에 등장 시점이 바뀐 동일한 순서에 대해 다른 특징을 추출하기 때문에 일반화가 어렵습니다.

* 분류 문제에서 CrossEntropy 대신 MSE를 사용하면 학습이 될까?
    * 확률분포를 가정하기 나름이라서 학습이 가능합니다. 
    * 예측값과 관측값의 차이가 클수록 Gradient가 커서 빠르게 학습되어야 하는데, Gradient가 오히려 작은 문제가 발생할 수 있습니다.
    * (Multi-Class Classification) MSE를 사용하면 클래스 간에 독립성이 추가적으로 가정되어야 하는데, 현실은 의존적이고 Softmax를 사용하면 의존적인 형태를 이루기 때문에 일반화는 상대적으로 어려울 것 같습니다.

* Residual Learning?
    * 레이어에서 출력과 입력 간의 차이를 학습하는 방법으로, 네트워크가 더 쉽게 학습할 수 있도록 합니다.
        * 레이어가 변화보단 변화량을 학습한다는 아이디어를 가지고 있음
    * Gradient Vanishing 문제 완화: Skip Connection을 통해 역전파를 전달하기 쉬워짐
    * 깊은 네트워크의 안정적인 학습 가능: Layer 출력의 표준편차가 줄어 안정적인 학습을 유도함

* Word2Vec의 원리는?
    * Word2Vec는 Word를 저차원의 Dense Vector로 변환시키는 방법입니다. 문맥상에 빈번하게 등장하는 단어 쌍끼리 유사도를 키우고, 이외에는 유사도를 낮추는 것이 기본 원리입니다. 
    * 특히, 유사도는 크게 거리와 방향의 두가지 개념이 있는데, 계산의 용이함 때문에 방향을 정량화 할 수 있는 내적(Inner Product)을 사용합니다. 
    * 즉, 빈번하게 등장하는 단어 쌍끼리 방향을 Align 시키는 방법으로 저차원의 벡터를 취득하는 방법이 Word2Vec 입니다.

* 분산학습(Distributed Learning)?
    * 분산학습(Distributed Learning): 여러 개의 GPU에 나누어 학습하는 방법입니다.
    * Data Parallelism: 동일한 모델을 여러 장치에 할당하고, 각 장치에서 다른 데이터를 사용하여 병렬로 훈련하는 방식
        * 데이터가 클수록 더 많은 장치를 활용하면 되기 때문에 확장성이 뛰어남
        * 각 장치에서 계산된 그래디언트를 집계하기 위해 통신 오버헤드가 존재함
        * 모델이 매우 크면 메모리 제한에 부딪힘
    * Model Parallelism: 모델을 여러 부분으로 나누어 각 부분을 다른 장치에 할당하여 병렬로 훈련하는 방식
        * 매우 큰 모델도 메모리 제한 없이 학습 가능함
        * 병목현상이 발생할 수 있고, 구현이 어려움