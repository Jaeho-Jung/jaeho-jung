# [1편] RNN으로 재즈 솔로 생성하기

> 이 글은 "Jazz is a Language" 시리즈의 1편이다.
> 7-피처 표현 위에서 Vanilla RNN을 NumPy로 직접 구현하고, PyTorch 버전과 비교한다.
> 코드 참조는 [Jazz-Is-a-Language](https://github.com/Jaeho-Jung/Jazz-Is-a-Language) 저장소의 `파일경로:줄번호` 형식으로 표기한다.

---

## RNN이 필요한 이유

재즈 솔로는 본질적으로 순서가 있는 데이터다. 지금 연주하는 음은 앞의 음과 연결되어 있다. 앞 구절에서 긴장을 만들고, 다음 구절에서 해소한다. 이를 음악적 언어로 "프레이징"이이라고 한다. 다시말해 재즈 즉흥 솔로는 음악적 맥락에 의존적이다.

이런 시퀀스 의존성을 처리하는 가장 직관적인 구조가 RNN이다.

피드포워드 네트워크는 각 타임스텝을 독립적으로 처리하기 때문에, `t-1` 타임스텝의 정보가 `t`의 예측에 전혀 반영되지 않는다. RNN은 **hidden state** `h_t`를 도입해 이 문제를 해결한다.

---

## 핵심 수식

Vanilla RNN의 타밍스텝 업데이트는 한 줄이다:

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
```

- `x_t`: 현재 타임스텝 입력 (임베딩 벡터)
- `h_{t-1}`: 직전 hidden state
- `W_xh`: 입력 가중치
- `W_hh`: 순환 가중치
- `tanh`: (-1, 1)로 squash

예측은 시퀀스 전체를 처리한 뒤 마지막 hidden state `h_T`에서 출력 헤드를 통해 생성된다:

```
pitch_logits  = W_pitch  · h_T + b_pitch   # (129 클래스)
dur_logits    = W_dur    · h_T + b_dur     # (동적 vocab)
```

---

# 아키텍처 구성

```
입력 피처 (7개 카테고리) → 임베딩 → Concat
                                       ↓
                              RNN (seq_len 타임스텝)
                                       ↓
                              마지막 hidden state h_T
                             ↙                    ↘
                   Pitch Head                 Duration Head
                  (129 classes)               (dynamic vocab)
```

임베딩 구성:

| 피처 | vocab | embed dim |
|------|-------|-----------|
| pitch | 129 | 32 |
| rel_pitch | 13 | 8 |
| duration | dynamic | 8 |
| prev_interval | 25 | 8 |
| chord_root | 13 | 8 |
| chord_quality | 7 | 4 |
| metric_pos | 48 | 8 |
| **합계** | | **76** |

---

## NumPy 구현: BPTT 직접 유도

PyTorch를 쓰면 `loss.backward()` 한 줄로 끝나지만, 그러면 실제로 무슨 일이 일어나는지 모른다. NumPy로 역전파를 직접 구현하면 각 단계의 gradient flow를 체감할 수 있다.

### Forward (RNNCell)

```python
def forward(self, x_t, h_prev):
    z_t = x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_H
    h_t = tanh(z_t)
    self.cache = {'x_t": x_t, "h_prev': h_prev, 'z_t': z_t, 'h_t': h_t}
    return h_t
```

### Backward (단일 타임스텝) - `src/RNN_numpy/layers/rnn.py:82`

```
h_t = tanh(z_t)  이므로
∂L/∂z_t = ∂L/∂h_t ⊙ (1 - h_t²)      ← tanh 미분
∂L/∂W_xh = (∂L/∂z_t)ᵀ · x_t
∂L/∂W_hh = (∂L/∂z_t)ᵀ · h_{t-1}
∂L/∂h_{t-1} = ∂L/∂z_t · W_hh         ← 다음 타임스텝으로 전달
```

코드:

```python
def backward(self, grad_h_t):
    h_t = self.cache['h_t']
    x_t = self.cache['x_t']
    h_prev = self.cache['h_prev']

    grad_z_t = grad_h_t * (1 - h_t ** 2)    # tanh 미분

    self.grad_W_xh = grad_z_t.T @ x_t
    self.grad_W_hh = grad_z_t.T @ h_prev
    self.gard_b_h  = grad_z_t.sum(axis=0)

    grad_x_t    = grad_z_t @ self.W_xh      # 임베딩으로 전달
    grad_h_prev = grad_z_t @ self.W_hh      # 이전 타임스텝으로 전달
    return grad_x_t, grad_h_prev
```

### BPTT: 시퀀스 역방향 루프 — `src/RNN_numpy/layers/rnn.py:213–216`

```python
def backward(self, grad_h_seq):
    # grad_h_seq: (batch, seq_len, hidden_size)
    # 마지막 타임스텝부터 역방향으로
    grad_h_t = np.zeros_like(grad_h_seq[:, 0, :])

    for t  in reversed(range(seq_len)):
        grad_h_t += grad_h_seq[:, t, :]     # 현재 timestep gradient 합산
        grad_x_t, grad_h_t = self.cell.backward(grad_h_t)
        grad_x_seq.insert(0, grad_x_t)

    return np.stack(grad_x_seq, axis=1)
```

### 가장 어려웠던 부분: gradient

구현에서 가장 까다로웠던 것은 수식 유도 자체보다는 타임스텝별 gradient 합산의 회계 처리였다. 각 타임스텝 t의 hidden state `h_t`는 두 곳에서 gradient를 받는다.

1. 해당 시점의 출력 loss에서 직접 내려오는 gradient: `grad_h_seq[:, t, :]`
2. 미래 타임스텝에서 역전파되어 온 gradient: 직전 루프의 `cell.backward`가 변환한 `grad_h_t`

`grad_h_t += grad_h_seq[:, t, :]` (`rnn.py:215`)의 `+=`가 위 두 경로의 합산이다. 이 한 줄의 위치를 잘못 잡으면 수렴은 하지만 틀린 gradient로 수렴하는 버그가 발생한다. 학습은 되는데 성능만 미묘하게 나쁜, 불친절하고 잡기 어려운 종류의 버그가 된다.

그래서 검증 장치가 필요하다.

---

## Gradient 검증: 수치 미분

- `src/RNN_numpy/tests/test_rnn.py:18–56`.

직접 유도한 analytical gradient가 맞는지 어떻게 확신할 수 있을까. 이 프로젝트에서는 수치 미분(numerical gradient check)을 테스트 코드로 작성했다.

원리는 central difference이다. 파라미터의 각 원소를 ±ε(1e-5)만큼 흔들어

```
numerical_grad = (L(θ + ε) − L(θ − ε)) / 2ε
```

를 계산하고, backward가 산출한 analytical gradient와 상대 오차를 비교한다. (`test_rnn.py:164, 199`)

```python
rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
```

분모에 두 gradient의 절대값 합을 사용하여 gradient크기가 0에 가까운 원소에서 절대 오차만 보면 통과하고 상대 오차만 보면 폭발하는 양극단을 피한다.

수치 미분은 파라미터 원소 하나마다 forward를 두 번 돌리므로 매우 느리다. 그래서 작은 텐서에서 레이어 단위로만 수정하였고, 모델 전체는 "loss가 수렴하는가 + 생성 결과가 합리적인가"로 간접 검증했다.

- 한계점: cross-check를 수행하지 않았다. Numpy 구현과 PyTorch 구현의 수치적 동등성 증명을 위해서는 동일 가중치를 NumPy 레이어와 PyTorch 레이어에 복사해 넣고 forward/backward 출력을 직접 대조하는 cross-check 단계가 필요하다. 수치 미분으로 1차 검증을 하였지만, 두 구현의 수치적 동등성과 대칭성 문제가 남는다.

---

## PyTorch 구현과의 차이

PyTorch 버전에서 초기화 방식을 보자. `src/RNN_pytorch/model.py:86–88`

```python
def _init_weights(self):
    for name, param in self.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
```

순환 가중치 `W_hh`에 직교 초기화(Orthogonal Initialization)를 쓰는 이유: 직교 행렬은 벡터의 크기(norm)를 보존하기 때문에 `h_t`에 `W_hh`를 반복 곱하더라도 gradient가 폭발하거나 소실되지 않는다. Vanishing gradient 문제를 초기화 수중에서 일부 완화하는 기법이다.

### 학습 안정화 정리

- Gradient clipping은 예방적으로 걸어두었다. `clip_grad_norm_(max_norm=1.0)` (`src/RNN_pytorch/trainer.py:68`). 학습 중 gradient exploding이 실제로 관측된 적은 없다. 문제가 발생해서 넣은 대응책이 아닌 표준적 예방 조치다.
- Numpy 구현에는 clipping이 없다. Numpy 버전은 역전파 메커니즘 이해가 목적인 최소 구현이기 때문에 의도적으로 제외했다.
- 실제로 마주한 문제는 vanishing 쪽이었고, 초기화로 완화를 시도하되 근본적으로는 구조적 해결로 넘어간다.(다음 편의 LSTM 구현)

---

## Vanilla RNN의 한계: Vanishing Gradient

BPTT를 T 타임스텝 역방향으로 수행할 때, gradient는 각 타임스텝마다 `W_hh`를 한 번씩 곱해 전달된다.

```
∂L/∂h_0 ∝ (W_hh)^T · ∂L/∂h_T
```

`W_hh`의 최대 singular value가 1보다 작으면 T가 커질수록 gradient가 기하급수적으로 줄어든다. 게다가 스텝 tanh의 미분 `(1 - h_t²) ≤ 1`이 추가로 곱해진다. 이 누적 곱은 고정된 행렬과 활성함수 미분의 반복이기 때문에 건너뛸 수 없다. 구조 자체의 한계다.

재즈 솔로에서 모티프(motif)는 중요한 음악적 요소로, 이전에 연주된 패턴을 이후 구간에서 변형하거나 재사용하는 특징을 가진다. Vanilla RNN은 이러한 장기 의존성을 포착하는 데 한계가 있다.

이 문제를 해결하기 위해 다음 편에서 LSTM을 도입한다.

---

## 회고

**배운 것**

BPTT를 직접 구현하며 gradient flow의 물리적 의미를 이해하게 됐다. 구현한 코드가 맞다는 것을 수치 미분으로 검증하는 습관이 NumPy 구현에서 얻은 가장 실용적인 수확이다.

**아쉬운 점**

RNN_numpy와 RNN_pytorch의 임베딩 피처 수가 다르다. 초기 구현에서 `chord_root_rel`, `next_chord_*` 등의 피처를 포함했는데, 이후 실험에서 7개로 정리했다. 두 구현의 비대칭성, 그리고 둘 사이의 수치적 동등성을 cross-check하지 않은 것이 아쉬움으로 남는다.

**다음 편 예고**

LSTM은 4개의 게이트를 도입해 무엇을 기억하고, 무엇을 지울지를 명시적으로 학습한다. gradient가 과거로 이동할 때 곱해지는 인자가 무엇으로 바뀌는지가 핵심이다. NumPy로 4개 게이트의 역전파를 유도하며 그 차이를 확인한다.

---

*다음: [2편] LSTM — 게이팅 메커니즘과 장기 의존성*