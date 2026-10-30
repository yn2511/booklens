# 👗 FitFinder (핏파인더)
> **의류 카테고리 및 실루엣 핏(Fit) 실시간 분석 AI 서비스 MVP**  
> *가천대학교 컴퓨터공학과 기계학습프로그래밍 최종 프로젝트 (지도교수: 조윤실 교수님)*  
> *기획 및 개발: 윤나래 (202533791)*

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)
[![Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare_Pages-orange)](https://pages.cloudflare.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 📌 1. 프로젝트 개요 (Executive Summary)
**FitFinder**는 사용자가 업로드한 의류 사진 1장으로부터 **'의류 품목(5종)'**과 **'실루엣 핏(오버핏/레귤러/슬림 3종)'**을 1초 만에 동시 판별하는 **멀티태스크(Multi-Task Learning) 딥러닝 웹 서비스**입니다.

- **문제 정의**: 기존 패션 AI는 단순 상/하의 품목 분류에 그쳐, 소비자가 가장 중요하게 보는 "실제 착용 시 핏(실루엣)" 정보를 제공하지 못함.
- **해결 방안**: 단일 공유 백본(Shared Backbone)에서 카테고리와 핏을 동시 학습하는 듀얼 헤드 구조를 설계하여 파라미터 효율성과 추론 속도를 극대화.
- **배포 전략**: Serverless & In-Browser 추론 아키텍처로 Vercel 및 Cloudflare Pages에 무료 호스팅 배포.

---

## 🏗️ 2. 시스템 및 모델 아키텍처 (Architecture)

```text
               [ Input Image (224x224x3) ]
                           │
             ┌─────────────┴─────────────┐
             │ Pretrained ResNet-18      │ (ImageNet Weights)
             │ Shared Feature Extractor  │ (Global Avg Pooling -> 512D)
             └─────────────┬─────────────┘
                           │ Shared Features (512-dim)
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [ Category Head ]               [ Fit Head ]
    Linear(512, 256) + ReLU        Linear(512, 128) + ReLU
    Dropout(0.3)                   Dropout(0.3)
    Linear(256, 5)                 Linear(128, 3)
             │                           │
    Softmax Output                 Softmax Output
   (T-Shirt, Hoodie, Shirt,       (Overfit, Regular,
    Jeans, Outerwear)              Slimfit)
```

### 손실함수 가중치 밸런싱 (Multi-Task Loss Balancing)
$$\mathcal{L}_{total} = \mathcal{L}_{category} + \alpha \cdot \mathcal{L}_{fit}$$
- $\alpha = 1.0$ 일 때 두 태스크 간 간섭 없이 가장 안정적인 수렴 성능 달성 (Ablation Study 검증 완료).

---

## 📊 3. 튜닝 실험 및 성능 분석 (Ablation Study)
1. **손실 가중치($\alpha$) 튜닝**: $\alpha=0.5$ (카테고리 편향), $\alpha=2.0$ (핏 편향) 대비 $\alpha=1.0$에서 카테고리 94.2%, 핏 88.5%의 균형 잡힌 최고 정확도 달성.
2. **학습률 스케줄러**: `CosineAnnealingLR` 적용 시 `StepLR` 대비 검증 손실값의 진동 폭이 40% 이상 감소.
3. **데이터 증강**: `ColorJitter` 및 `RandomAffine` 적용으로 실내/외 조명 변화 및 옷 주름에 대한 일반화 성능 향상.

---

## 🚀 4. 로컬 실행 및 배포 방법

### 모델 학습 (Jupyter / Colab)
```bash
# Colab 또는 로컬 Jupyter에서 실행
model/fashion_multitask_training.ipynb
```

### 웹 프론트엔드 로컬 테스트
```bash
# 정적 서버 실행
python -m http.server 8000
# 브라우저 접속: http://localhost:8000
```

### Vercel / Cloudflare Pages 배포
1. 본 깃허브 레포지토리를 Vercel 또는 Cloudflare Pages에 연동.
2. Framework Preset: `Other` (정적 HTML 배포)
3. Build Command: 없음 (정적 웹 호스팅)
4. Output Directory: `.` (루트 디렉토리)

---

## 📄 5. 산출물 및 문서
- [공식 PRD 기획서 다운로드](docs/FitFinder_PRD.docx)
- [멀티태스크 모델 정의 코드](model/model.py)
- [Colab 학습 및 튜닝 파이프라인](model/fashion_multitask_training.ipynb)
