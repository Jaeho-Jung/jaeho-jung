# [2편] LSTM — 게이팅 메커니즘과 장기 의존성

> 이 글은 "Jazz is a Language" 시리즈의 2편이다.
> LSTM이 RNN의 Vanishing Gradient 문제를 어떻게 해결하는지,
> 그리고 NumPy로 4개 게이트의 역전파를 직접 유도한 과정을 다룬다.
> 코드 참조: [Jazz-Is-a-Language](https://github.com/Jaeho-Jung/Jazz-Is-a-Language) 저장소, `파일경로:줄번호` 형식.

---

## 1편 복습: RNN의 한계

1편에서 Vanilla RNN을 구현했다. 핵심은 이 수식이었다:

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
```

문제는 BPTT 과정에서 gradient가 시간을 거슬러 올라가면서 `W_hh`를 반복 곱한다는 것이다:

```
∂L/∂h_0 ∝ (W_hh)^T  ×  (W_hh)^T  × ... × ∂L/∂h_T
```

`W_hh`의 singular value가 1보다 조금이라도 작으면, 멀리 있는 타임스텝의 gradient는 0으로 수렴한다. 재즈 솔로처럼 수십 타임스텝에 걸친 패턴은 RNN이 학습하기 어렵다.

---

## LSTM의 아이디어: Cell State와 게이트

LSTM(Long Short-Term Memory)은 1997년 Hochreiter & Schmidhuber가 제안한 구조다. 핵심 아이디어는 두 가지다:

1. **Cell State `c_t`**: hidden state와 별도로 유지되는 "장기 메모리". 게이트를 통해서만 수정되므로 gradient가 비교적 깨끗하게 흐른다.
2. **게이트(Gate)**: sigmoid로 0~1 사이 값을 생성해 정보의 흐름을 조절한다.

```
           정보를 얼마나 지울까      새 정보를 얼마나 쓸까
             Forget Gate            Input Gate
                  ↓                      ↓
           c_t = f_t ⊙ c_{t-1}     +    i_t ⊙ g_t
                         ↑                     ↑
                   이전 cell state        후보 cell state
      
           h_t = o_t ⊙ tanh(c_t)
                  ↑
              Output Gate
```

---

## 핵심 수식

```
# 게이트 전처리 (입력 + hidden 선형 변환)
z_ifo   = x_t · U_ifo.T  + h_{t-1} · W_ifo.T  + b_ifo
z_cell  = x_t · U_cell.T + h_{t-1} · W_cell.T + b_cell

# 게이트 활성화
i_t, f_t, o_t = split(sigmoid(z_ifo), 3)   # Input / Forget / Output
g_t           = tanh(z_cell)                # Candidate cell state

# Cell state 업데이트
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

# Hidden state
h_t = o_t ⊙ tanh(c_t)
```

구현에서는 i, f, o 게이트를 하나의 행렬 `U_ifo` (3×hidden_size, input_size)로 묶어서 처리했다. forward 시 `np.split(..., 3)`으로 분리한다.

---

## NumPy 역전파

RNN의 역전파는 한 줄(`∂tanh`)이었지만, LSTM은 4개 경로를 동시에 추적해야 한다.
셀 단위 backward는 `src/LSTM_numpy/layers/lstm.py:96`부터다.

### Cell State Gradient 합산 — `lstm.py:127–134`

```python
# h_t = o_t * tanh(c_t) 이므로
grad_c_from_h = grad_h_t * o_t * (1 - tanh_c_t ** 2)    # lstm.py:127

# 두 곳에서 c_t로 gradient가 들어온다:
# 1) 현재 h_t 계산 경로
# 2) 다음 타임스텝에서 전달된 grad_c_t (c_{t+1} = f_{t+1} * c_t + ...)
grad_c_total = grad_c_from_h + grad_c_t                  # lstm.py:130

# 이전 cell state로 전달
grad_c_prev = grad_c_total * f_t                         # lstm.py:134
```

- `grad_c_total = grad_c_from_h + grad_c_t`의 덧셈을 두고 "LSTM은 덧셈이라 vanishing에 강하다"고 흔히 말하지만, 정확하게 따지면 두 가지 질적 변화가 동시에 일어난다.

1. 분기가 보존된다.

forward에서 `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`가 덧셈 구조이기 때문에, 미분의 합 규칙에 의해 backward에서도 cell state로 들어오는 gradient는 (현재 h_t 경로 + 미래 타임스텝 경로)의 합니다. 한쪽 경로가 죽어도 다른 쪽 경로가 살아있으면 gradient는 전달ㄷ된다.

2. 과거로 가는 곱셈 인자의 정체가 바뀐다.

한 스텝 과거로 갈 때 곱해지는 것이 RNN처럼 고정 행렬이 아니라, 입력에 따라 매 스텝 달라지는 게이트 값 `f_t`다. f_t는 학습되는 함수이므로 모델이 정보를 유지해야 한다고 판단한 차원에서는 f_t ≈ 1 을 출력해 gradient를 거의 무손실로 통과시킨다. 행렬곱이 아닌 대각(원소별) 곱이라 차원간 혼합도 없다.

따라서 RNN에서의 고정 행렬의 비선형 반복곱이, 덧셈으로 분기가 보존되는 동시에 학습 가능한 게이트의 원소별 곱으로 바뀌었다.

### 각 게이트의 Gradient — `lstm.py:137–140`

```python
# sigmoid 미분: σ'(z) = σ(z)(1 - σ(z))
grad_i_t = grad_c_total * g_t  * i_t * (1 - i_t)        # Input gate
grad_f_t = grad_c_total * c_prev * f_t * (1 - f_t)      # Forget gate
grad_o_t = grad_h_t * tanh_c_t * o_t * (1 - o_t)        # Output gate
grad_g_t = grad_c_total * i_t  * (1 - g_t**2)           # Candidate (tanh 미분)
```

### 파라미터 Gradient — `lstm.py:144–160`

```python
grad_z_t = np.hstack([grad_i_t, grad_f_t, grad_o_t])   # (batch, 3*hidden)

self.grad_U_ifo  = grad_z_t.T @ x_t                    # 입력 가중치
self.grad_W_ifo  = grad_z_t.T @ h_prev                 # 순환 가중치
self.grad_U_cell = grad_g_t.T @ x_t
self.grad_W_cell = grad_g_t.T @ h_prev

# 이전 타임스텝으로 전달
grad_h_prev = grad_z_t @ self.W_ifo + grad_g_t @ self.W_cell   # lstm.py:155
grad_x_t    = grad_z_t @ self.U_ifo + grad_g_t @ self.U_cell

return grad_x_t, grad_h_prev, grad_c_prev               # lstm.py:160
```

LSTM 역전파에서 주의할 점: `grad_h_prev`와 `grad_c_prev`를 동시에 이전 타임스텝으로 넘겨야 한다. RNN의 루프에서는 `grad_h_t` 하나만 반환시켰지만 LSTM에서는 두 state의 gradient를 함께 순환시킨다.

### 검증


1편과 동일하게 수치 미분 기반 gradient check를 테스트로 작성했다 (`src/LSTM_numpy/tests/test_lstm.py:19`의 `numerical_gradient`). 게이트가 4개로 늘어나면 손으로 유도한 수식에서 부호나 인자 하나가 틀리기 쉬운데(특히 `grad_f_t`의 `c_prev`와 `grad_i_t`의 `g_t`를 바꿔 쓰는 실수), central difference 대조를 통해 이런 실수를 기계적으로 거른다.

---

## Forget Gate

Forget Gate의 역할을 직관적으로 이해하기 위해 음악적으로 생각하자면, "지금 코드가 바뀌었으니 이전 코드 정보는 잊어라"의 역할을 한다.

- `f_t = 0`: "이전 기억을 지워라" — 코드 전환 시점
- `f_t = 1`: "이전 기억을 유지해라" — 코드 지속 시점


Input Gate는 반대로 "지금 이 음이 얼마나 중요한가"를 결정한다. 박자상 중요한 위치(1박, 3박)에서 더 높은 input gate 값이 나올 것으로 기대할 수 있다.

학습 후 실제로 그렇게 동작하는지 gate activation을 시각화해보는 것도 흥미로운 후속 실험이 될 것이다.

---

## RNN vs LSTM 비교


| 항목 | RNN | LSTM |
|------|-----|------|
| 파라미터 수 | `hidden² + hidden×input` | `~4 × (hidden² + hidden×input)` |
| 과거로의 곱셈 인자 | 고정 행렬 `W_hh` (반복 행렬곱) | 학습되는 게이트 `f_t` (원소별 곱) |
| Gradient 분기 | 단일 경로 | cell/hidden 두 경로, 덧셈으로 합산 |
| 장기 의존성 | 약함 (seq_len > ~20) | 상대적으로 강함 |
| 역전파 복잡도 | 단순 | 4개 게이트 + 2개 state 동시 추적 |

파라미터가 약 4배 많아지지만, 그에 비해 성능 향상이 항상 극적이지는 않다. 실제로 이 프로젝트에서도 생성 샘플의 청취 기준으로 LSTM이 RNN보다 확연히 낫다고 말하기 어려웠다. 관찰과 해석은 3편 회고에서 정리한다.

학습 설정은 RNN과 동일하게 유지했다. `W_hh`(PyTorch의 `weight_hh`) 직교 초기화(`src/LSTM_pytorch/model.py:85–87`), 예방적 gradient clipping `max_norm=1.0`(`src/LSTM_pytorch/trainer.py:68`). 1편과 마찬가지로 exploding은 관측되지 않았다.

---

## 회고

**배운 것**

`grad_c_total = grad_c_from_h + grad_c_t`와 `grad_c_prev = grad_c_total * f_t`, 이 두 줄이 "왜 LSTM이 vanishing gradient에 강한가"의 핵심이다. 수식을 읽을 때는 당연해 보였던 것이 직접 구현해보고 나서야 곱셈 인자를 모델이 제어한다는 말의 의미를 체감했다.


**구현 관련 주의**

LSTM_numpy의 가중치 이름이 직관적이지 않다. `U_ifo` (입력 가중치), `W_ifo` (순환 가중치)로 분리했는데, PyTorch의 `weight_ih`, `weight_hh` 컨벤션과 대응된다. 

**다음 편 예고**

LSTM도 결국 타임스텝을 순차적으로 처리한다. 즉, t번째 hidden state를 계산하려면 반드시 t-1번째가 끝나야 한다. Transformer의 Self-Attention은 모든 타임스텝을 병렬로 처리하고, 임의의 두 위치간 직접 연결을 만듦으로써 이 제약을 완전히 제거한다.

---

*다음: [3편] Transformer — Self-Attention*
