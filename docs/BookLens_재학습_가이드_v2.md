# BookLens v2 재학습 실행 가이드

> **역할: 검증용 재학습 (배포 모델 아님).** 실제 booklens.cloud 배포 모델은 v1 (Test 97.33%, `models/booklens_resnet18.onnx`, 21권 임베딩 `data/catalog_embeddings.json`도 v1 산출물)입니다. 이 가이드의 v2 (Test 96.00%, 6에폭 조기종료)는 Val Loss·Train Acc·조기종료 로직 검증을 위한 재학습 기록입니다.

## v1과 달라진 점 (교수님 피드백 대응)
- **Val Loss·Train Acc 기록**: v1은 Train Loss / Val Acc / Val F1만 기록했지만, v2는 5개 지표(Train Loss, Train Acc, Val Loss, Val Acc, Val Macro-F1)를 매 에포크 출력합니다.
- **조기종료(Early Stopping)**: Val F1이 3회 연속 개선되지 않으면 자동 종료합니다 (`PATIENCE = 3`). 최대 에포크는 15로 늘렸습니다.
- **곡선 자동 시각화**: 학습이 끝나면 Train/Val Loss, Train/Val Acc, Val F1 곡선이 모델별로 자동 생성됩니다. Train-Val 곡선을 나란히 비교해 과적합 여부를 정량으로 확인할 수 있습니다.
- 그 외(데이터 로딩, 분할, 증강, 모델 정의, 테스트 평가, Grad-CAM, ONNX 변환)는 v1과 동일합니다.

## 실행 방법 (Colab)
1. `BookLens_재학습_v2.ipynb` 파일을 Colab에서 엽니다 (colab.research.google.com → 파일 → 노트북 업로드, 또는 드라이브에 저장 후 열기).
2. 상단 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 인지 확인합니다.
3. **런타임 → 모두 실행**을 누릅니다.

## 데이터 경로 확인
- 2번 셀이 구글 드라이브를 마운트하고 `book_dataset` 폴더를 자동으로 찾습니다.
  - 후보 순서: `기계학습프로그래밍 8주차 강의자료/과제3/book_dataset` → `과제3/book_dataset` → `MyDrive/book_dataset` → (못 찾으면 드라이브 전체 스캔)
- `FileNotFoundError`가 나오면 드라이브 안에 `book_dataset/picture_book`, `text_book`, `comic_book` 세 폴더가 있는지 확인해 주세요.

## 예상 소요 시간
- v1 실측 기준: Baseline 357.9초 + ResNet-18 547.5초 ≈ **15분** (고정 8에포크 기준)
- v2는 조기종료로 ResNet-18은 더 일찍 끝날 가능성이 높고, Baseline은 최대 15에포크까지 갈 수 있어 **학습 15~25분**, Grad-CAM·ONNX 변환까지 포함하면 **전체 약 25~30분** 예상됩니다.

## 실행 후 확인할 것
- 에포크 로그에 `Val Loss`, `Train Acc`가 함께 출력되는지
- 조기종료 메시지(몇 에포크에서 멈췄는지)
- 자동 생성된 곡선 2장(모델별 3패널)
- 테스트 지표·혼동행렬이 새 값으로 갱신되는지

## 주의
- 시드는 고정되어 있지만 GPU 연산 특성상 v1과 소수점 수준의 차이가 날 수 있습니다.
- 새 지표가 확정되면 발표 자료의 수치와 비교가 필요합니다 (덱 반영은 별도 진행).
- 실행 결과 로그와 곡선을 보내주시면 분석 문구도 새 수치로 갱신해 드립니다.
