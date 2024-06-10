# Effective Detection of Generated Images Using Fast Fourier Transform and Discrete Cosine Transform

다양한 전처리 방법과 푸리에 변환을 이용한 효과적인 생성 이미지 탐지

### KDFS 2023 동계학술대회 논문 개제 실험

#### 해상도 변환 방법에 따른 성능 변화

| Transform                        | Train Accuracy | Valid Accuracy | Test Accuracy       |
| -------------------------------- | -------------- | -------------- | ------------------- |
| Resizing                         | 0.9919         | 0.9758         | 0.7641              |
| Minimal Resizing & Random Crop   | 0.9969         | 0.9992         | 0.8684 (+0.1043)    |
| Zero-Padding & Random Crop       | 0.9966         | 0.9982         | 0.8660 (+0.1019)    |
| Self-replication & Random Crop   | 0.9961         | 0.9985         | 0.8664 (+0.1023)    |

#### 고속 푸리에 변환 적용 유무에 따른 성능 변화

| Transform   | Train Accuracy | Valid Accuracy | Test Accuracy    |
| ----------- | -------------- | -------------- | ---------------- |
| No Applied  | 0.9991         | 0.9997         | 0.7594           |
| FFT Applied | 0.9992         | 0.9989         | 0.8614 (+0.0973) |

#### 이산 코사인 변환 적용 유무에 따른 성능 변화

| Transform   | Train Accuracy | Valid Accuracy | Test Accuracy    |
| ----------- | -------------- | -------------- | ---------------- |
| No Applied  | 0.9991         | 0.9997         | 0.7594           |
| DCT Applied | 0.9970         | 0.9994         | 0.9328 (+0.1687) |
