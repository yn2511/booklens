# BookLens AI: 실시간 도서 형태 분류 및 NLP 도서 큐레이션 서비스 MVP

- **과목명**: 기계학습프로그래밍 (조윤실 교수님)
- **소속**: 가천대학교 IT융합대학 컴퓨터공학과 2학년
- **학번 / 성명**: 202533791 / 윤나래
- **라이브 웹 서비스 MVP**: [https://booklens.cloud](https://booklens.cloud)

---

## 1. 프로젝트 개요

BookLens AI는 9주차 수업 지침인 **"기계학습 모델 학습에 그치지 않고 실제 동작하는 서비스 MVP(Minimum Viable Product)를 도출하라"**는 목표에 따라 기획·구현된 초등 도서관 AI 북 도슨트 시스템입니다.

정적인 웹 크롤링 표지 데이터셋과 달리, 사용자가 도서관 서가에서 스마트폰 카메라로 책을 비출 때 필연적으로 마주치는 **기하학적 원근 왜곡(Perspective Distortion)**, **천장 형광등 조명 반사(Lighting Reflection)**, **손목 기울어짐(Rotation)**이라는 실환경 도메인 갭(Domain Gap)을 전이학습(Transfer Learning)과 단계별 데이터 증강(Ablation Study)으로 극복하였습니다.

분류된 도서 카테고리는 9주차 실습에서 학습한 **TF-IDF 벡터화 및 코사인 유사도(Cosine Similarity)** 엔진과 결합되어, 독자의 관심사에 맞춘 추천 도서 3선과 맞춤형 해설 카드를 실시간 웹 인터페이스로 제공합니다.

---

## 2. 1~9주차 강의 커리큘럼 연계표

본 프로젝트는 수업 외적인 외부 기법을 배제하고, 수업 교안과 실습에서 배운 머신러닝·딥러닝 기법들을 하나의 유기적인 파이프라인으로 구현하였습니다.

| 주차 | 수업 주제 | 본 프로젝트 구현 및 적용 내용 |
| :--- | :--- | :--- |
| **1주차** | 기계학습 개요 및 확률 | 입력(X)과 출력(Y) 매핑 정의, Softmax 확률 해석, 재현성을 위한 시드(SEED=42) 고정 |
| **2주차** | 데이터 전처리 및 탐색 | 클래스별 이미지 기초 통계(종횡비, RGB 평균 밝기, 명암 대비) 추출 및 손상 데이터 검증 |
| **4주차** | 분류 모델 평가 지표 | 다중 클래스 불균형 방지를 위한 Macro-F1 기준 모델 가중치 선택 및 Confusion Matrix 정밀 분석 |
| **5주차** | 비지도학습 및 차원축소 | ResNet-18 512차원 특징 벡터를 PCA 및 t-SNE로 2차원 축소하여 전이학습 전·후 군집 분리도 검증 |
| **6주차** | 신경망 원리 및 튜닝 | PyTorch 학습 루프 구축, CrossEntropyLoss, AdamW 최적화, CosineAnnealingLR 스케줄러 적용 |
| **8주차** | CNN, 전이학습, XAI | Scratch CNN vs ResNet-18 Feature Extractor vs Fine-Tuning 비교, 4단계 증강 Ablation, layer4 Grad-CAM 및 Sobel 3x3 엣지 필터 오답 원인 규명 |
| **9주차** | 통계적 NLP 및 서비스 MVP | 도서 코퍼스 대상 TfidfVectorizer + cosine_similarity 기반 맞춤 도서 추천, 라이브 웹 서비스 MVP(https://booklens.cloud) 배포 |

---

## 3. 실험 결과 요약 (Ablation Study)

동일한 초기 난수 시드(`SEED=42`)와 Stratified 분할(Train 180장, Val 90장, Test 90장) 환경에서 통제 실험을 수행한 결과입니다.

### 3.1 전이학습 전략 및 학습률 스케줄러 종합 성능

| 실험 번호 | 모델 및 학습 전략 | 데이터 증강 (Augmentation) | 학습 파라미터 수 | Test Loss | Test Accuracy | Test Macro-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 1** | Scratch (small_cnn) | Baseline (Resize + Flip) | 380,355개 | 0.8921 | 68.89% | 0.6742 |
| **Exp 2** | Feature Extractor (백본 동결) | Baseline (Resize + Flip) | 1,539개 | 0.4120 | 88.89% | 0.8845 |
| **Exp 3** | Fine-Tuning (전체 미세조정) | Baseline (Resize + Flip) | 11,178,051개 | 0.2854 | 92.22% | 0.9180 |
| **Exp 4** | Fine-Tuning | + 원근 왜곡 단독 (RandomPerspective 0.35) | 11,178,051개 | 0.2310 | 94.44% | 0.9421 |
| **Exp 5** | Fine-Tuning | + 조명 반사 단독 (ColorJitter 0.3) | 11,178,051개 | 0.2541 | 93.33% | 0.9312 |
| **Exp 6** | **[최종] Fine-Tuning + CosineLR** | **복합 실환경 증강 (원근+조명+회전)** | 11,178,051개 | **0.1412** | **96.67%** | **0.9655** |

### 3.2 데이터 증강(Augmentation) 기법별 정량 기여도

- **Baseline Fine-Tuning**: 92.22% (기준선)
- **+ 원근 왜곡(RandomPerspective 0.35) 단독**: 94.44% (+2.22%p) -> 비스듬한 촬영 각도로 인한 사다리꼴 외곽선 왜곡 극복
- **+ 조명 반사(ColorJitter 0.3) 단독**: 93.33% (+1.11%p) -> 도서관 천장 형광등 반사광 및 그림자 대응
- **[최종] 복합 실환경 증강 + CosineAnnealingLR**: 96.67% (+4.45%p) -> 복합 실환경 노이즈 극복 및 후반부 손실 진동 억제

---

## 4. 시스템 아키텍처 및 서비스 MVP (booklens.cloud)

```
[사용자 모바일/PC 웹 브라우저 (https://booklens.cloud)]
                      |
                      v 1) 실시간 카메라 캡처 (WebRTC / Canvas)
[클라이언트 전처리 엔진]
  - 8주차 1교시 원리: 3x3 Sobel 엣지 필터로 텍스트 획 밀도 실시간 계산
  - 왜곡 보정: 프레임 가이드라인을 통한 원근 왜곡 1차 억제
                      |
                      v 2) 224x224 RGB 텐서 변환 및 추론 요청
[서버리스 추론 파이프라인]
  - 백본: ResNet-18 미세조정 가중치 (booklens_resnet18_best.pt)
  - 출력: 클래스별 Softmax 확률 벡터 (picture_book, text_book, comic_book)
                      |
                      v 3) 예측 클래스 + Softmax Margin + 엣지 밀도 결합
[9주차 NLP 기반 도서 큐레이션 엔진]
  - 9주차 1교시 TF-IDF 벡터화 및 코사인 유사도(cosine_similarity) 매칭
  - 독자 연령층 맞춤형 추천 도서 3선 카드 및 리포트 실시간 렌더링
                      |
                      v 4) UI 피드백 출력
[최종 웹 화면: 도서 카테고리 태그 + 신뢰도 게이지 + 추천 도서 3권 큐레이션 카드]
```

---

## 5. 프로젝트 디렉토리 구조

```
BookLens-AI/
├── README.md                                 # 프로젝트 개요 및 실험 결과 보고서
├── requirements.txt                          # 의존성 패키지 목록
├── BookLens_AI_최종프로젝트.ipynb             # 전체 실험 및 시각화 주피터 노트북
├── src/
│   ├── model.py                              # 모델 정의 (ResNet-18 전이학습 및 small_cnn)
│   ├── curator.py                            # 9주차 TF-IDF 코사인 유사도 큐레이션 엔진
│   └── pipeline.py                           # 이미지 전처리 및 추론 파이프라인
└── weights/
    └── booklens_resnet18_best.pt             # 최고 성능 모델 가중치 (Exp 6)
```

---

## 6. 설치 및 실행 방법

### 6.1 의존성 설치
```bash
pip install -r requirements.txt
```

### 6.2 모델 추론 및 큐레이션 파이프라인 실행
```python
from src.pipeline import BookLensPipeline

pipeline = BookLensPipeline(weights_path="weights/booklens_resnet18_best.pt")
result = pipeline.predict_and_curate("sample_cover.jpg")

print("예측 카테고리:", result["predicted_class"])
print("신뢰도:", result["confidence"])
print(result["curation_report"])
```
