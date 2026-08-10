# [3편] Transformer — Self-Attention

> 이 글은 "Jazz is a Language" 시리즈의 3편이다.
> GPT-style Decoder-Only Transformer를 PyTorch와 NumPy로 구현하고,
> RNN/LSTM과의 근본적인 구조 차이를 다룬다.
> 코드 참조: [Jazz-Is-a-Language](https://github.com/Jaeho-Jung/Jazz-Is-a-Language) 저장소, `파일경로:줄번호` 형식.

---

## RNN/LSTM의 구조적 제약

LSTM은 vanishing gradient 문제를 많이 완화했지만, 순차 의존성이라는 근본적인 제약이 남아 있다. `h_t`를 계산하려면 반드시 `h_{t-1}`이 먼저 완료되어야 한다. 이 순차 의존성으로 인해 두 가지 문제점이 발생한다.

1. 병렬화 불가: 시퀀스 길이에 비례해 순차 계산이 필요하다. GPU의 병렬 연산을 활용하지 못한다.
2. 고정된 병목: 모든 과거 정보가 고정 크기의 hidden state `h_t` 하나에 압축된다. seq_len이 길면 초기 정보가 손실된다.

Transformer는 이 두 제약을 모두 제거한다.

---

## Self-Attention의 직관

Self-Attention의 핵심 아이디어는 시퀀스의 모든 위치의 직접 연결하는 것이다. 

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

- Q(Query): "나는 무엇을 찾고 있는가?"
- K (Key): "나는 어떤 정보를 갖고 있는가?"
- V (Value): "내가 실제로 전달하는 값"
- √d_k: attention score의 분산이 지나치게 커지는 것을 방지하는 스케일링

Causal (Decoder-Only) 설정에서는 위치 `t`가 `t+1, t+2, ...`를 볼 수 없다. 하삼각 마스크를 적용해 미래를 차단한다.

---

## Teacher Forcing

RNN/LSTM은 마지막 hidden state `h_T`만 출력 헤드에 통과시킨다. 이번 Transformer에서는 모든 위치에서 동시에 다음 토큰을 예측하도록 구현하였다.

이로 인해 학습 효율이 크게 향상된다. 길이 T의 시퀀스 하나로 T개의 예측-정답 쌍을 동시에 학습할 수 있다.


코드에서는 Dataset이 이렇게 처리한다 (`src/Transformer_pytorch/dataset.py:108–118`):

```python
features = {k: arr[start:end]       for k in feature_keys}   # (T,)
targets  = {'pitch': pitch[start+1:end+1],                    # (T,) — 한 칸 shift
            'duration': dur[start+1:end+1]}
```

모델은 `(B, T, vocab)` shape logit을 반환하고, loss는 `(B*T,)` 토큰 전체의 평균이다.

이 학습방식은 teacher forcing이다. 위치 t의 입력은 항상 ground truth `x_t`이고, 학습 중 모델은 단 한 번도 자기 출력 위에서 예측해본 적이 없다. 이 받식은 inference에서 exposure bias를 만들 수 있다. 4편에서 두 표현 방식을 대조하며 다룬다.

---

## 아키텍처 구성

```
입력 피처 (7개) → 임베딩 → Concat (B, T, 76)
                              ↓
                    input_proj Linear (76 → 128)
                              ↓
              ┌───────────────────────────────┐
              │  Pre-LN TransformerBlock × 3  │
              │                               │
              │  x = x + MHA(LN(x))           │
              │  x = x + FFN(LN(x))           │
              └───────────────────────────────┘
                              ↓
                     Pitch Head (B, T, 129)
                     Duration Head (B, T, vocab_dur)
```

하이퍼파리미터 (Transformer_pytorch 기준, vocab/임베딩 정의는 `src/Transformer_pytorch/config.py:28–51`):

| 항목 | 값 |
|------|----|
| d_model | 128 |
| n_heads | 4 |
| num_layers | 3 |
| dropout | 0.3 |
| seq_len | 64 |
| label smoothing | 0.1 |
| LR schedule | Linear warmup + CosineAnnealingWarmRestarts |

작은 모델 + 강한 정규화(dropout 0.3, label smoothing)는 WJD의 데이터 규모에 맞춘 선택이다.

mixed precision(AMP) 사용 시 gradient clipping은 unscale을 먼저 한 뒤 적용한다. (`src/Transformer_pytorch/trainer.py:122–124`). GreadScaler가 곱해둔 스케일이 남아있는 상태에서 norm을 재면 clipping 기준이 왜곡되기 때문이다. 1, 2편과 마찬가지로 clipping은 예방 조치이며 exploding은 관측되지 않았다.

---

## Pre-LN vs Post-LN

원래 "Attention is All You Need" (2017)의 Transformer는 Post-LN이다:

```
# Post-LN
x = LN(x + MHA(x))
x = LN(x + FFN(x))
```

GPT-2 이후 실무에서는 Pre-LN이 표준이 됐다:

```
# Pre-LN
x = x + MHA(LN(x))
x = x + FFN(LN(x))
```

Pre-LN의 장점: 학습 초기 gradient가 안정적이다. LN이 잔차 연결 전에 적용되므로, gradient가 LayerNorm을 우회해 residual path로 직접 흐를 수 있다. 결과적으로 warmup 없이도 학습이 시작되고, 더 높은 학습률을 쓸 수 있다.

---

## Multi-Head Attention 역전파

Multi-Head Attention 역전파가 이 프로젝트에서 가장 까다로운 부분이었다. 가장 중요한 코드는 다음 한 줄이다.

```python
# attn: (B, h, T, T) — post-softmax attention weights
# grad_attn: upstream gradient
grad_scores = attn * (grad_attn - (grad_attn * attn).sum(axis=-1, keepdims=True))
```

여기서 `grad_scores`는 softmax 이전 값, 즉 attention score에 대한 gradient다. 다시 말해 forward에서 다음과 같은 흐름이 있었다면,

```
scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores)
out = attn @ V
```

위 한 줄은 역전파 중에서

```
grad_attn -> grad_scores
```

로 넘어가는 부분이다. 겉으로 보면 단순한 원소별 곱과 합산처럼 보이지만, 실제로는 softmax의 Jacobian-vector product를 압축한 형태다.

### 왜 `grad_attn * attn`이 아닐까?

ReLU나 tanh같은 활성함수는 각 원소가 독립적으로 변한다. 입력 `x_i`가 바뀌면 출력 `y_i`만 바뀌고, 다른 출력에는 직접 영향을 주지 않는다. 그래서 역전파도 보통 원소별 미분을 곱하면 끝난다.

하지만 softmax는 다른다.

```
a_i = exp(z_i) / Σ exp(z_j)
```

각 출력 `a_i`는 자기 입력 `z_i`만 보는 것이 아니라, 분모를 통해 모든 `z_j`를 함께 본다. 따라서 `z_j` 하나가 변하면 softmax 출력 전체가 바뀐다. attention에서 한 query가 보는 분포 하나를 생각하면, 길이 T짜리 벡터에 대해 T × T Jacobian이 생긴다.

성분별 미분은 다음과 같다.

```
∂a_i/∂z_j = a_i(δ_ij − a_j)
```

이를 행렬로 쓰면 softmax Jacobian은 다음 형태가 된다.

```
J = diag(a) − a aᵀ
```

즉 softmax의 역전파는 본질적으로 “원소별 곱”이 아니라 “Jacobian과 gradient 벡터의 곱”이다.

### 하지만 Jacobian을 직접 만들면 안 된다.

attention weight attn의 shape은 보통 `(B, h, T, T)`이다.

여기서 마지막 축 T에 대해 softmax가 적용된다. 즉 각 batch, 각 head, 각 query position마다 길이 T짜리 softmax 분포가 하나씩 있다.

만약 각 softmax 분포마다 Jacobian을 직접 만들면, 원래 `(B, h, T, T)`였던 attention 텐서가 역전파 과정에서 `(B, h, T, T, T)`로 커진다.

마지막에 Jacobian의 T × T가 추가되기 때문이다. sequence length가 조금만 커져도 메모리 비용이 급격히 증가한다. 그래서 실제 구현에서는 Jacobian을 명시적으로 만들지 않고, Jacobian-vector product를 닫힌 형태로 계산해야 한다.

### Softmax 역전파의 닫힌 형태

softmax 출력이 `a`, upstream gradient가 `g`라고 하자. 우리가 구하려는 것은 softmax 입력 `z`에 대한 gradient다.

```
∂L/∂z = Jᵀ g
```

softmax Jacobian은 대칭이므로 다음처럼 쓸 수 있다.

```
∂L/∂z = (diag(a) − a aᵀ) g
```

이를 전개하면,

```
= diag(a)g − a(aᵀg)
= a ⊙ g − a ⊙ ⟨g, a⟩
= a ⊙ (g − ⟨g, a⟩)
```

이 식이 그대로 코드가 된다.

```
grad_scores = attn * (grad_attn - (grad_attn * attn).sum(axis=-1, keepdims=True))
```

```
(grad_attn * attn).sum(axis=-1, keepdims=True)
```

이 부분은 마지막 축 기준으로 grad_attn과 attn의 행별 내적을 계산한다. 수식으로는 `⟨g, a⟩`에 해당한다.

즉 각 attention row마다 upstream gradient의 weighted average를 구하는 것이다.

```
grad_attn - weighted_average
```

는 upstream gradient에서 그 행의 평균적인 gradient 방향을 빼는 연산이다.

```
attn * (...)
```

를 통해 softmax 출력값 a를 다시 곱한다. 결과적으로 각 원소는 다음 형태가 된다.

```
a_i (g_i − Σ_j g_j a_j)
```

이것이 softmax 입력 score에 대한 gradient다.

### 직관

이 식을 직관적으로 보면 더 이해하기 쉽다. softmax 출력은 확률분포처럼 합이 항상 1이다.

```
Σ a_i = 1
```

따라서 모든 성분을 똑같이 키우는 방향의 gradient는 softmax 공간에서 의미가 없다. 어떤 값을 키우려면 상대적으로 다른 값을 줄여야 한다. softmax는 절대적인 크기보다 성분 간 상대적 차이에 반응하기 때문이다.

그래서 역전파에서도 단순히 grad_attn을 그대로 전달하지 않는다. 먼저 행별 가중 평균 `⟨g, a⟩`을 빼서, softmax 분포 안에서 실제로 의미 있는 상대적 gradient만 남긴다.

즉 이 코드는 다음을 한 번에 수행한다.

1. upstream gradient에서
2. softmax 분포의 평균 방향을 제거하고,
3. 각 attention weight의 크기만큼 다시 스케일링한다.

그래서 최종 구현은 Jacobian을 만들지 않으면서도 softmax의 full Jacobian 효과를 정확히 반영한다.

```
grad_scores = attn * (grad_attn - (grad_attn * attn).sum(axis=-1, keepdims=True))
```

겉으로는 NumPy broadcasting 한 줄이지만, 안쪽에서는 `diag(a) − a aᵀ`라는 softmax Jacobian 전체를 암묵적으로 계산하고 있다.이 한 줄이 MHA 역전파에서 메모리 효율성과 수학적 정확성을 동시에 만족시키는 핵심이다.

### Multi-Head Attention Backward

```python
# 1. out_proj backaward
grad_out = self.out_proj.backward(grad_out) # (B, T, C)

# 2. heads로 reshape
grad_heads = grad_out.reshape(B, T, h, d).transpose(0, 2, 1, 3) # (B, h, T, d)

# 3. attn_weights @ V 분리
grad_v    = attn.T @ grad_heads # (B, h, T, d)
grad_attn = grad_heads @ v.T    # (B, h, T, T)

# 4. dropout backward
grad_attn = self.attn_dropout.backward(grad_attn)

# 5. softmax backward
grad_scores = attn * (grad_attn - (grad_attn * attn).sum(axis=1, keepdims=True))

# 6. causal mask
grad_scores *= causal_mask[:T, :T]

# 7. scale
grad_scores *= head_dim ** -0.5

# 8. Q·K^T 분리
grad_q = grad_scores @ K    # (B, h, T, d)
grad_ = grad_scores.transpose(0, 1, 3, 2) @ Q # (B, h, T, d)

# 9. reshape + proj backward
return q_proj.backward(grad_q) + k_proj.backward(grad_k) + v_proj.backward(grad_v)
```

- causal mask: foward에서 -inf 마스킹된 위치는 softmax 출력이 0이므로 수학적으로는 gradient도 0이 되지만, 수치 안정성을 위해 명시적으로 마스킹하였다.

실제 구현 시 어려웠던 점은 4차원 텐서의 transpose 축 관리였다.

### Pre-LN TransformerBlock Backward

Pre-LN 구조의 역전파는 잔차 연결 덕에 깔끔하다.

```python
def backward(self, grad_output):
    # 두 번째 잔차 분기
    grad_x1 = grad_output + self.ln2.backward(self.ffn.backward(grad_output))

    return grad_x1 + self.ln1.backward(self.mha.backward(grad_x1))
```

잔차 연결 `x = x + f(LN(x))`의 backward는 항상 `grad += branch_grad` 형태다. gradient가 residual shortcut + 서브레이어의 두 경로로 분기되어 흐른다.

---

## 구현 중 발생한 버그들

NumPy로 Transformer를 구현할 때 여러 버그들이 많았다. 수치 미분 gradient check에 걸려 발련된 세 가지를 버그를 대표적으로 기술하겠다.

1. `Linear.backward`의 3D 텐서 처리

기존 `linear.py`는 2D 입력만 가정한다.

```python
self.grad_W = grad_output.T @ self._x
```

3D 텐서 (B, T, out)가 입력되면 .T가 축을 모두 뒤집다. `(out, T, B) @ (B, T, in)`가 되기 때문에 오류가 발생한다.

수정본

```python
flat_g = grad_output.reshape(-1, self.output_features)  # (B*T, out)
flat_x = self._x.reshape(-1, self.input_features)       # (B*T, in)
self.grad_W = flat_g.T @ flat_x                         # (out, in)
```

2. `FeedForward.backward`의 순서 오류

Forward: `fc1 → GELU → Dropout → fc2`

기존 backward (잘못된 순서)

```python
grad = fc2.backward(grad)
grad = grad * gelu_derivative(pre_act)
grad = dropout.backward(grad)   
```

Dropout이 fc2 다음에 와야 하는데 GELU 뒤에 있다.

수정본

```python
grad = fc2.backward(grad)
grad = dropout.backward(grad)
grad = grad * gelu_derivative(pre_act)
return fc1.backward(grad)
```

3. `TransformerBlock.backward`의 잘못된 gradient 합산

초기 구현에서 forward에서 캐시한 출력값인 `self._ffn_out` 을 gradient에 더하는 오류가 있었다. 잔차 연결의 backward에서는 출력값을 더하는 것이 아니라, gradient를 두 경로로 분기해 합산하는 것이다.

---

## RNN/LSTM과의 차이

| 항목 | RNN / LSTM | Transformer |
|------|-----------|-------------|
| 처리 방식 | 순차 (t-1 → t) | 병렬 (모든 위치 동시) |
| 위치 간 거리 | O(T) 단계 | O(1) 직접 연결 |
| 정보 병목 | hidden state 크기 | 없음 (전체 시퀀스 attention) |
| 학습 시 target | 마지막 위치만 | 모든 위치 (T배 효율) |
| 파라미터 | 비교적 작음 | 비교적 큼 |
| 추론 시 | stateful | 매번 전체 컨텍스트 재계산 |

---

## 세 모델 비교 결과에 대한 기록
여기까지 읽었다면 자연스러운 질문이 따라온다. "그래서 Transformer가 RNN보다 얼마나 좋았는데?"

솔직하게 말하면, 자신 있게 답할 수 없다. 두 가지 이유 때문이다.

첫째, 비교 가능한 정량 기록이 없다. 각 트레이너는 train/val loss를 출력하고 best model을 저장하지만(`src/Transformer_pytorch/trainer.py:72-74`), 세 모델을 동일 프로토콜(동일 split, 동일 평가 스크립트)로 비교하는 실험을 체계적으로 수행·기록하지 못했다. 0편에서 "통제된 ablation이 가능하도록 설계했다"라고 작성한 것은 사실이지만, 그 설계를 실험으로 완성하지는 못했다.

둘째, 청취 기준으로는 차이가 명확하지 않다. 생성 샘플을 들어봤을 때 LSTM이나 Transformer가 RNN보다 확연히 낫다고 말하기 어려웠다. 의외의 결과가 나왔고, 더 생각하게 되었다. 가능한 해석은 두 가지다.

- 표현이 이미 너무 많은 일을 해준다. 7-피처 표현은 화성 컨텍스트(chord_root/quality), 박절 위치(metric_pos), 직전 선율 음정(prev_interval)을 매 토큰에 부착한다. 다음 음 하나를 그럴듯하게 예측하는 데 필요한 정보의 대부분이 직전 몇 토큰의 피처 안에 국소적으로 존재한다. 그렇다면 장기 의존성 처리 능력이 발휘될 여지 자체가 줄어든다. 아키텍처의 차이가 드러나려면 표현이 덜 친절해야 한다는 가설이며, 이것이 4편에서 표현 자체를 바꿔보는 동기 중 하나가 된다.
- 평가 해상도가 부족하다. 사람 귀의 단발 청취로는 "국소적 자연스러움"과 "장기 구조"를 분리해서 듣기 어렵다.
차이가 정말 없었던 것인지, 차이를 측정할 도구가 없었던 것인지 현재로선 구분할 수 없다.

두 해석 모두 같은 결론을 가리킨다. 다음에 가장 먼저 해야 할 일은 모델 추가가 아니라 평가 지표 구축이다. 이 논의는 4편의 섹션에서 이어진다.

---

## 회고

**배운 것**

Sofmax backward의 `attn * (grad - (grad * attn).sum(-1, keepdims=True))`를 처음 봤을 때 당황스러웠다. Jacobian을 `a_i(δ_ij - a_j)`로 직접 전개하고 JVP로 정리하면 자연스럽게 나오는 수식이지만, Jacobian을 만들지 않고 Jacobian-벡터 곱만 계산한다는 발상이 역전파 구현의 핵심이라는 것을 배웠다.

**한계**

Positional Embedding의 부재. PyTorch 버전은 학습된 위치 임베딩을 더하지만, Numpy 버전은 `input_proj`만 있다. 솔로에서 마디 내 위치는 `metric_pos` 피처로 이미 인코딩되어 있어 영향이 제한적이지만, 이론상 누락이다.

그리고 위에 적었듯, 세 모델에 대한 체계적 정량 비교를 완료하지 못하였다.

---

*다음: [4편] Music Transformer + Modified REMI — 이벤트 토큰 표현과 Relative Attention*