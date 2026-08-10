# [4편] Music Transformer + Modified REMI

> 이 글은 "Jazz is a Language" 시리즈의 4편이자 마지막 편이다.
> 3편까지의 7-피처 테이블형 표현과 달리, 이번 편은
> 음악을 이벤트 토큰 시퀀스로 표현한다. 
> 여기서 쓰는 이벤트 표현("Modified REMI")은
> [Wu & Yang, "The Jazz Transformer on the Front Line" (ISMIR 2020)](https://arxiv.org/abs/2008.01307)이
> WJazzD에 맞춰 설계한 스킴을 그대로 채택한 것이고, 이 프로젝트가 한 일은
> 그 표현을 [Music Transformer (Huang et al., 2018)](https://arxiv.org/abs/1809.04281)의
> relative self-attention 아키텍처 위에서 직접 구현·학습시킨 것이다.
> 두 접근의 차이와 각각의 장단점을 정리한다.
> 코드 참조: [MusicTransformer-Pytorch](https://github.com/Jaeho-Jung/MusicTransformer-Pytorch) 저장소, `파일경로:줄번호` 형식.
 
---

## 파이프라인의 차이

1~3편의 세 모델(RNN/LSTM/Transformer)과 이번 편의 Music Transformer는 별개의 두 파이프라인이다. 같은 WJD를 다르게 처리한다.

| 구분       | 파이프라인 A — 테이블형                      | 파이프라인 B — Modified REMI        |
| -------- | ----------------------------------- | ------------------------------ |
| 사용 모델    | RNN / LSTM / Transformer            | Music Transformer              |
| 코드 위치    | `Jazz-Is-a-Language/src/Preprocess` | `MusicTransformer-Pytorch/src` |
| 변환 과정    | WJD → 7-피처 DataFrame                | WJD `mcsv` → REMI 토큰           |
| 데이터 형태   | 표 형태의 feature row                   | 토큰 시퀀스                         |
| 시간 격자    | 마디당 48-grid                         | 마디당 64슬롯 / 박당 16슬롯             |
| 코드 표현    | `root` / `quality` 정수 인덱스           | `Tone` / `Type` / `Slash` 토큰   |
| Velocity | 없음                                  | 있음                             |
| 입력 단위    | 한 시점의 feature 묶음                    | 하나의 음악 이벤트 토큰                  |


따라서 이번 편의 비교는 같은 입력에 모델만 바꾼 ablation이 아니라, 데이터 표현과 아키텍처를 묶음으로 선택한 두 설계 철학의 비교다. 

---

## 지금까지의 접근: 테이블형 피처 표현

RNN, LSTM, Transformer가 사용한 표현은 전저리 결과가 "음표 하나 = DataFrame의 한 행"인 표이고, 각 행이 7개의 이산 정수 인덱스 열을 갖는 구조다.

```
각 음표 = 7개 정수 인덱스의 조합
{ pitch: 60, rel_pitch: 7, duration: 12, prev_interval: 14,
  chord_root: 0, chord_quality: 1, metric_pos: 24 }
```

모델 입력에서는 각 열이 독립된 vocab과 임베딩 테이블을 갖는다. 7개 임베딩을 concat한 76차원 벡터가 음표 하나의 표현이 된다. 데이터셋 레벨에서는 `dataset.py:59–73`이 결측을 처리하며 정수 배열을 만들고, `:108–118`에서 한 칸 shift된 (입력, 타킷) 쌍을 구성한다.

특징

- 명시적: 화음, 박자 위치, 선율 간격이 별도 채널로 분리되어 있다
- 1:1 대응: 음표 하나 = 토큰 하나. 시퀀스 길이 = 음표 개수
- 조건의 상시 부착: 코드 컨텍스트(chord_root/quality)가 모든 토큰에 피처로 붙어있다
- 전처리 의존: 화음 분석, 박자 격자 정렬이 선행되어야 한다

모델은 다음 음표의 pitch와 duration을 예측하는 태스크를 푼다.

---

# REMI: 이벤트 토큰 표현

[REMI (REvamped MIdi-derived events)](https://arxiv.org/abs/2002.00212)는 피아노 롤이나 테이블이 아닌, 이벤트 시퀀스로 음악을 표현하는 방식이다. 원래 팝 피아노 생성 논문(Pop Music Transformer, 2020)에서 제안되었다.

### 기존 MIDI-like 토큰화와의 차이

기존 MIDI-like 이벤트 토큰화(Performance encoding: `NOTE_ON / NOTE_OFF / TIME_SHIFT / VELOCITY`)는 시간을 연속적 time-shift의 누적으로 표현한다. 문제는 모델이 지금 마디의 몇 번째 박인지 알기 위해서는 앞선 time-shift들을 전부 더애햐 한다는 것이다.

REMI는 이를 명시적 격자로 바꾼다. Bar 토큰이 마디 경계를, Position 토큰이 마디 내 위치를 직접 알려주고, Note Duration이 NOTE_OFF를 대체, Chord·Tempo 토큰이 화성·템포를 명시한다.

재즈 프레이징은 박절 위치에 강하게 종속된다. 강박의 코드톤, 약박의 논-코트톤, 싱코페이션 모두 마디 내 어디인가가 중요하다. Bar/Position이 명시되면 모델이 이 박절-화성 관계를 time-shift 누적 없이 직접 attention할 수 있다. 뒤에서 다룰 relative attention과 결합하면 "정확히 한 마디 전"과 같은 거리 참조가 안정적으로 가능해진다.

```
Bar → Position(0) → Chord(Cm7) → Pitch(60) → Duration(4) →
      Position(12) → Pitch(64) → Duration(2) → ...
      Position(24) → Chord(F7) → Pitch(65) → Duration(8) → ...
Bar → Position(0) → ...
```

각 토큰은 하나의 사건을 나타내는 정수 인덱스다. 음표 하나가 아닌 이벤트 하나가 하나의 토큰이므로, 같은 음표를 표현하는 데 여러 토큰이 필요하다.

---

## Modified REMI

원본 REMI는 팝 피아노용으로 설계되어 재즈 솔로에 필요한 화음 체계와 구조 정보를 다루지 않는다. 이 간극을 WjazzD에 맞춰 메운 것이 Jazz Transformer 논문의 Section 3.2, "Data Representation"이다. 이 프로젝트가 쓰는 "Modified REMI"의 토큰 스키마는 이 절을 그대로 채택한 것이다.

원 논문은 이벤트를 4개의 카테고리로 나눈다: 
- note-related: NOTE-VELOCITY, NOTE-ON, NOTE-DURATION
- metric-related: BAR, POSITION, TEMPO-CLASS, TEMPO
- chord-related: CHORD-TONE, CHORD-TYPE, CHORD-SLASH
- structure-related: PHRASE,q MLU, PART, REPITITION

`event_to_encodings`, `convert_to_remi.py:223`
```
beat  → [Part-End?/Part-Start?] [Bar?] [Position] [Tempo-Class] [Tempo]
chord → [Position?] [Chord-Tone] [Chord-Type] [Chord-Slash]
note  → [Position?] [Phrase?] [MLU tokens?] [Note-Velocity] [Note-On] [Note-Duration]
```

### 코드 토큰의 분해

원 논문에서 코드를 분해 토큰으로 표현하는 방식을 제시한다. 예를 들면, "Bbmaj7"을 `Chord-Tone_Bb` + `Chord-Type_maj7` + `Chord-Slash_*`로 분해하는 식이다. 원 논문은 WJD에 등장하는 418개의 고유 코드 표기를 그대로 토큰화하면 그중 69%가 5개 미만 솔로에서만 등장해 학습이 어렵다고 지적하고, 루트·퀄리티·베이스음을 분리해 71개의 코드 관련 이벤트로 압축하는 방법을 제안한다. 이 프로젝트의 chords decomposition은 이 해법을 구현한 것이다. 루트와 퀄리티가 독립적으로 일반화된다는 이점 또한 존재한다.

참고로 이전 파이프라인(테이블형) 또한 코드를 root와 quality로 분해하였지만, 베이스음(슬래시 코드)은 분리하지 않았다. Chord-Slash를 별도 축으로 둔 것은 Modified REMI에서만 있는 세분화다.

### 구조 태그와 MLU

구조 태그: form의 섹션 경계를 `Part-Start`/`Part-End`, 반복 구조를 `Rep-Start`/`Rep-End` 토큰으로 표시한다.(`:245-252`)

MLU (Melodic-Level Unit) 태그: WJD가 제공하는 선율 구조 레이블(Phrase, 반복, 변형 패턴 등)을 토큰으로 인코딩한다.(`mlu_processor.py`)
원 논문은 이 구조 이벤트의 유무로 두 모델을 학습해 objective metric으로 비교했고, 구조 이벤트를 포함한 쪽이 단·중기 구조성 지표에서 더 나았다고 보고한다. 이 결과는 3편에서 다룬 관찰—7-피처 테이블형 표현의 구조 정보 부재로 어떤 아키텍처도 장기 구조를 배울 수 없었다—과 같은 결론을 가리킨다.

```python
class MLUTag:
    is_phrase: bool          # 구절의 시작점인가
    has_void: bool           # 빈 공간(쉼표 구간)이 있는가
    rep_backref: bool        # 이전 구절을 참조하는 반복인가
    rep_variation: bool      # 변형 반복인가
    typ: str                 # MLU 타입 (예: 'lick', 'run', ...)
    sub_typ: str
```

음표 하나를 나타내는 데 Position + Velocity + Note-On + Duration + MLU + 구절 정보가 필요하기 때문에, 같은 솔로를 인코딩하면 테이블형보다 시퀀스가 훨씬 길어진다.

### 코드 재삽입 주기

원 논문에서 벗어나 이 프로젝트가 다르게 결정한 지점이 하나 있다. 원 논문은 "POSITION 이벤트는 음표 시작·코드 변화·템포 변화가 있을 때만 발생한다"고 명시한다—즉 코드 토큰은 변화 시점에만 삽입된다. 하지만 순수하게 "변화 시점에만" 삽입하면 한 코드가 여러 마디 지속될 때 코드 정보가 컨텍스트에서 너무 멀어진다는 문제가 있다. 그래서 이 프로젝트는 긴 코드를 4박마다 주기적으로 재삽입하도록 바꿨다 (`repeat_long_chord=True, repeat_beats=4`, `collect_chords`, `:84–116, :95`).

이 선택은 인과적 조건화라는 REMI 계열 표현의 일반적 특성과 맞물린다. 코드 토큰이 해당 구간의 음표 토큰보다 시퀀스상 먼저 등장하므로, causal attention 하에서 모델은 항상 "현재 코드를 본 상태에서" 음을 생성한다. 별도의 cross-attention이나 encoder 없이, decoder-only 구조만으로 조건부 생성이 구현된다. 다만 코드 토큰도 결국 시퀀스의 일부라서, 모델이 참조 관계를 학습해야 하고 극단적으로는 생성 시 입력 조건과 다른 코드를 스스로 출력해버릴 수도 있다. 반면 테이블형의 조건 주입은 더 강제적이다. 생성 시에도 코드 피처는 모델이 만드는 것이 아니라 입력된 코드 진행 타임라인에서 계산되어 매 스텝 주입되므로(`Jazz-Is-a-Language/src/Transformer_pytorch/generator.py:20–81`), 모델이 코드 진행에서 이탈하는 것이 원리적으로 불가능하다. 조건의 강제력 vs 표현의 일반성, 이 트레이드오프가 두 파이프라인의 본질적 차이다.

---

## 아키텍처

Jazz Transformer 논문은 자신들이 설계한 위 이벤트 표현을 Transformer-XL 위에서 학습시켰다. 이 프로젝트는 같은 표현을 Music Transformer(Huang et al., 2018) 위에서 학습시켰다. 즉 이 프로젝트의 실질적인 설계 기여는 Jazz Transformer의 표현과 Music Transformer의 아키텍처를 원 논문들에는 없던 조합으로 결합한 것이다.

두 아키텍처 모두 "긴 컨텍스트 + 상대 위치"를 다루지만 핵심 기여가 다르다.

- Transformer-XL: 핵심은 segement-level recurrence이다. 이전 세그먼트의 hidden state를 cache하여 다음 세그먼트에서 재사용함으로써, 학습 길이를 넘어서는 사실상 무한한 컨텍스트를 제공한다. Jazz Transformer가 이를 택한 이유도 솔로당 평균 3,000에 가까워 세그먼트 단위로 끊어 학습할 수밖에 없었기 때문이다. recurrence를 가능하게 하기 위한 부속 설계로 상대 위치 인코딩도 사용한다.
- Music Transformer: 핵심은 relative self-attention. 위치 쌍 (i, j)의 거리 i-j에 대한 임베딩을 attention score에 더해, "한 마디 전", "두 박 전"같은 거리 관계를 직접 학습힌다.

Music Transformer를 선택한 이유

1. WJD 솔로는 곡당 수백~수천 토큰 수준이라, seq_len을 충분히 잡으면 솔로의 상당 부분이 컨텍스트에 포함된다. Transformer-XL의 강점인 학습 길이 초과 컨텍스트가 필요하지 않다고 생각했다.
2. 음악의 모티프 반복, 전위는 절대 위치가 아닌 상대 거리의 함수이므로, 거리 자체를 모델링하는 relative attention이 문제 구조에 더 직접적으로 부합하다고 판단했다.
3. Jazz Transformer와 다른 아키텍처를 씀으로써, 같은 표현 위에서 아키텍처만 달리한 비교 지점이 생긴다.

이 비교는 논문 수준의 검토에 기반한 선택이고, Transformer-XL 베이스라인을 실제로 학습해 정량 비교하지는 않았다. 저장소의 transformer_xl/ 디렉토리는 포크에 포함된 코드일 뿐 학습에 사용하지 않았다. 두 접근의 실측 비교는 향후 과제로 남겨둔다.

---

## Music Transformer: Relative Self-Attention

원본 [Music Transformer (Huang et al., 2018)](https://arxiv.org/abs/1809.04281)의 핵심 기여는 **Relative Position Representation (RPR)**이다.

### Relative Attention

음악에서 절대 위치보다 "한 마디 전 모티프의 반복", "ii-V-I 진행에서 V에 도달한 시점"같은 상대 거리가 더 중요하다.

일반 Transformer의 Positional Embedding은 절대 위치를 인코딩한다. 두 가지 약점이 있다.

- 간접 학습: "거리 k만큼 떨어진 두 위치의 관계"를 위치별 임베딩들의 조합으로 간접 학습해야 한다.
- 위치 일반화 실패: 같은 프레이즈가 곡 앞에 나오느냐 뒤에 나오느냐에 따라 다르게 처리될 수 있고, 학습 길이보다 긴 시퀀스에는 learned PE는 정의될 수 없고, sinusodial도 외삽 성능이 떨어진다.

Relative Attention은 두 위치 `i`, `j` 사이의 거리 `i-j`에 대한 임베딩 `E_r`을 사용한다.

```
# 일반 Attention
score(i, j) = q_i · k_j / √d_k
 
# Relative Attention
score(i, j) = (q_i · k_j + q_i · E_{i−j}) / √d_k
```

두 번째 항 `q_i · E_{i−j}`가 "위치 i에서 위치 j를 바라볼 때, 그 거리가 얼마나 중요한가"를 학습한다. 거리당 임베딩 하나이다.

### Skew Algorithm

`E_{i−j}`를 나이브하게 구현하면, 모든 (i, j) 쌍의 임베딩을 담은 텐서 `R ∈ (T, T, D)`를 명시적으로 만들어 Q와 곱해야 하기 때문에 중간 메모리가 O(T²D)로 커진다. Skew 알고리즘은 이를 O(TD)로 줄인다. causal 마스킹 하에서 실제로 참조되는 거리가 T개뿐이므로 이 텐서 전체 대신 E ∈ (T, D) 하나면 충분하다는 것이다. 참고로 attention score 행렬 (T, T) 자체는 relative attention 여부와 무관하게 어차피 만들어야 하므로 절약 대상이 아니다. Skew가 절약하는 것은 `(T, T, D)`인 상대 임베딩 중간 텐서다.

실행 흐름을 `model/rpr.py`에서 줄 단위로 따라가면,

1. `:478` `_get_valid_embedding`: 학습된 최대 길이의 거리 임베딩 `Er`에서 현재 시퀀스 길이에 필요한 마지막 행들만 슬라이스 (`:515–527`).
2. `:480` `qe = torch.einsum("hld,md->hlm", q, rpr_mat)`: Q `(h, T, D)`와 E `(T, D)`를 곱해 `qe ∈ (h, T, T)`. `qe[l, m]` = "query l과 거리 임베딩 m의 내적". 문제: 우리가 원하는 `S_rel[i, j] = q_i · E_{i−j}`와 인덱싱이 어긋나 있다—qe의 열은 "거리 번호"인데 원하는 행렬의 열은 "절대 위치 j"라서, 행이 하나 내려갈 때마다 올바른 정렬이 한 칸씩 밀린다.
3. **`:537–539`**: triu 마스크(flip)로 각 행에서 causal하게 유효한 거리 항목만 남긴다.
4. **`:541` `F.pad(qe, (1,0, ...))`**: 마지막 축 왼쪽에 0 한 열 패딩 → `(h, T, T+1)`.
5. **`:542` `torch.reshape(..., (h, T+1, T))`**: 같은 메모리 버퍼를 행 길이 T로 재해석. 행 길이가 T+1에서 T로 바뀌면서 각 행의 시작점이 한 칸씩 "흘러내리고", 이것이 2번에서 필요했던 행별 한 칸 시프트를 **전 행에 동시에** 수행한다. reshape는 데이터 복사 없이 인덱스 재해석만 하므로 추가 메모리·연산이 사실상 0—이것이 트릭의 본체다.
6. **`:544` `srel = qe[:, 1:, :]`**: 패딩으로 생긴 첫 행을 버리면 올바르게 정렬된 `S_rel` 완성. **`:482–484`**에서 일반 attention score에 더해진다.

### attribution

이 프로젝트에서 코드를 작성한 층위를 구분하면 다음과 같다.

1. RPR/skew 코어(`model/rpr.py`): Damon Gwinn의 오픈소스 MusicTransformer-Pytorch를 포크해 사용했고, 해당 함수들의 docstring에 원저자가 명시되어 있다 (`:517, :530`). 차용한 코드도 그냥 쓰지 않고 줄 단위로 분석했으며, 위의 skew 설명이 그 분석의 결과물이다.
2. 이벤트 표현의 설계(vocab 스키마, 코드 분해, 구조/MLU 태그의 정의): Wu & Yang(2020)의 Jazz Transformer 논문 Section 3.2를 따랐다.

---

## Teacher Forcing과 Exposure Bias

두 파이프라인 모두 학습은 GPT-style next-token prediction, 즉 teacher forcing이다. 학습 중 모델의 입력은 항상 ground truth이고, 모델은 자기 출력 위에서 예측해본 적이 없다. 추론에서는 샘플링된 자기 출력이 다음 입력이 되므로(`generator.py:76–81`의 temperature 샘플링 + `multinomial`), 학습 분포와 추론 분포가 어긋난다. 어색한 토큰을 한 번 뽑으면 그 뒤의 모든 예측이 "본 적 없는 컨텍스트" 위에서 이뤄지고 오류가 복리로 누적된다. 이것이 exposure bias다.

흥미롭게도, 두 표현이 구조적으로 다른 취약성을 갖는다.

테이블형: 문법 오류가 구조적으로 불가능하다. 매 스텝이 반드시 (pitch, duration) 쌍을 출력하고, 조건 피처인 코드는 타임라인에서 강제 주입되며, 파생 피처(metric_pos, prev_interval, rel_pitch)는 생성된 음에서 결정론적으로 계산된다. 출력 형식이 깨질 방법이 없다. exposure bias는 오직 이상한 음/길이 선택 등 내용으로만 나타난다.

Modified REMI: 문법 자체가 학습 대상이다. 단일 토큰 스트림이므로 "Position 다음에 Note-On이 온다", "Note-On 뒤에 Duration이 따라온다" 같은 구문 규칙까지 모델이 학습해야 한다. exposure bias가 내용 오류뿐 아니라 구문 오류(Position 없는 Note-On, Duration 누락, 마디 내 Position 역행)로도 나타날 수 있고, 구문이 한 번 깨지면 이후 디코딩 전체가 오염된다.

이처럼 표현의 선택이 exposure bias의 표면적까지 결정한다. 테이블형은 문법을 하드코딩으로 보호하는 대신 유연성을 잃고, REMI 계열은 일반성을 얻는 대신 문법까지 학습해야 한다. 완화책으로 temperature 샘플링을 사용했고, top-k/top-p, 그리고 REMI 쪽에는 문법 제약 디코딩(현재 상태에서 허용되는 토큰 타입만 남기고 logit 마스킹)이 추후 구현 과제이다.

---

## 평가

Music Transformer의 학습은 NLL loss와 token accuracy로 추적했다 (`utilities/run_model.py:59–83`의 `eval_model`이 validation loss와 함께 argmax 일치율 기반 accuracy를 계산한다. `dataset/e_piano.py:139`의 `compute_epiano_accuracy`).

이것이 이 프로젝트에서 기록으로 남은 유일한 정량 지표다. 3편에서 적었듯 테이블형 세 모델은 비교 가능한 정량 기록이 없고, 강박에서 코드톤이 선택되는가, 프레이즈 단위 호흡이 생기는가, 모티프 변형이 나타나는가와 같은 음악적 평가는 청취 기반 정성 관찰에 머물렀다.

token accuracy는 다루기 쉽지만 음악적 품질의 대리 지표로는 약하다. 그럴듯한 다른 음을 모두 오답 처리하기 때문이다. Jazz Transformer(Wu & Yang, 2020)가 이미 이 문제를 인식하고 제안한 객관 지표 세트(pitch class entropy, grooving pattern similarity, chord progression irregularity, fitness scape plot 기반 structureness indicator, MIREX-like continuation prediction)를 도입하는 것이 다음 단계이다.

---

## 두 표현의 비교

| 항목 | 7-피처 테이블형 | Modified REMI |
|------|--------------|--------------|
| 음표당 토큰 수 | 1 | ~4–6 |
| 격자 해상도 | 마디당 48-grid | 마디당 64슬롯 (박당 16) |
| 화음 정보 | 매 토큰마다 피처로 부착 | 변화 시점 삽입 + 4박마다 재삽입 |
| 화음 표현 | root/quality 정수 인덱스 | Tone/Type/Slash 분해 토큰 |
| 생성 시 조건 | 타임라인에서 강제 주입 (이탈 불가) | 모델이 토큰으로 참조 (이탈 가능) |
| 구조 정보 | 없음 | MLU 태그, Part/Rep 경계 |
| Velocity | 미포함 | 포함 |
| 출력 문법 | 구조적으로 보장 | 학습 대상 (깨질 수 있음) |
| 시퀀스 길이 | 짧음 | 길음 |
| 전처리 | `Jazz-Is-a-Language/src/Preprocess` | `MusicTransformer-Pytorch/src` (별개) |
| 도메인 지식 | 7개 피처 설계에 반영 (직접 설계) | 토큰 사전 설계에 반영 (Wu & Yang 2020 채택) |
| 모델 부담 | 피처 의미를 주입해줌 | 토큰 관계를 처음부터 학습 |

- 테이블형: "중요한 피처가 무엇인지 안다"는 가정을 전제한다. pitch, chord_quality, metric_pos를 따로 주는 것은 모델에게 독립적으로 중요한 요소를 직접적으로 제공하는 것과 같다.
- REMI 계열: 모델이 중요한 관계를 Self-Attention으로 직접 학습힌다. 피처 엔지니어링 대신 더 많은 데이터와 모델 용량이 필요한다.

---

## 4개 모델 비교 요약

| 모델 | 표현 | 예측 방식 | 파라미터 수 | 정량 기록 |
|------|------|----------|-----------|-----------|
| RNN | 7-피처 테이블형 | 마지막 hidden state | 소 | 없음 |
| LSTM | 7-피처 테이블형 | 마지막 hidden state | 중 | 없음 |
| Transformer | 7-피처 테이블형 | 전체 위치 (GPT-style) | 중 | 없음 |
| Music Transformer | Modified REMI | 전체 위치 | 대 | loss + token acc |

---

## 한계

1. 체계적 정량 비교를 완료하지 못했다.
0편에서 "통제된 ablation이 가능하도록 설계했다"고 했지만, 세 테이블형 모델을 동일 프로토콜로 비교한 정량 결과가 기록으로 남아 있지 않다. 정량 지표가 남은 것은 Music Transformer의 loss/accuracy뿐이다. 설계와 실험 사이의 이 간극이 이 프로젝트의 가장 큰 미완이다.

2. 청취 기준으로 LSTM/Transformer가 RNN보다 낫다고 말하기 어려웠다. 3편 회고에서 자세히 다뤘듯, 두 해석이 가능하다.
2.1. 피처 표현이 예측에 필요한 정보를 매 토큰에 국소적으로 제공하므로, 장기 의존성 처리라는 상위 아키텍처의 강점이 발휘될 여지 자체가 작았다. 
2.2. 청취 기반 평가의 해상도가 차이를 분별하기에 부족했다. 어느 쪽인지 판정하려면 측정 도구가 필요하다.

3. 정성 관찰 기준, 공통된 약점이 존재했다.
3.1. 장기 구조 부재, 프레이즈 반복·변형·기승전결 없이 이어지는 라인, 리듬 단조화.
3.2. pitch/duration을 독립 헤드로 예측해 결합 분포가 약하고 고빈도 duration으로 수렴하는 경향.
전자는 표현에 구조 정보가 없는 것이 원인이므로 Jazz Transformer가 제안한 MLU/구조 토큰을 채택해 대응했지만, 그 토큰들이 이 프로젝트의 파이프라인에서도 실제로 기여했는지는 ablation으로 검증하지 못했다.

4. 데이터 규모.
처음부터 WJD가 약 450개로 부족한 것을 고려하여 설계했다. 12키 증강, 도메인 지식을 압축한 피처/토큰 설계, 작은 모델 + 강한 정규화 등. REMI처럼 일반적인 표현일수록 데이터 요구량이 커지므로, 작은 WJD 위에서는 테이블형의 강한 inductive bias가 유리한 출발점일 가능성이 높다. 대규모 MIDI 코퍼스 사전학습 후 WJD 파인튜닝하는 옵션도 있다. REMI 계열 표현은 사전학습과의 호환성이 좋다는 점이 REMI 선택에 힘을 싣는다.

### 발전 방향

1. 평가 지표 구축
Jazz Transformer가 제안한 pitch class histogram entropy, grooving pattern similarity, chord progression irregularity 등. 정량적 측정 없이는 위 2번의 차이가 없었다는 주장과 이후 개선을 입증할 수 없을 것이다.
2. 4개 모델의 동일 프로토콜 정량 비교
3. MLU/구조 토큰 ablation
Jazz Transformer의 Model A/B 비교를 이 파이프라인에서도 재현해, 추가 토큰이 생성 품질에 기여하는지 검증.
4. 문법 제약 디코딩(REMI의 구문 오류를 디코딩 단계에서 차단)
5. 사전학습 + 파인튜닝
6. 데모 배포

---

## 회고

**시리즈를 마치며**

이 프로젝트를 통해 배운 가장 중요한 것은 각 선택의 trade-off를 직접 체감하는 것이었다. BPTT를 구현했을 때, LSTM의 cell state 경로가 왜 vanishing gradient에 강한지 이해됐다. GPT-style 전체 위치 예측을 구현했을 때, 마지막 hidden state만 쓰는 것이 얼마나 비효율적인지 실감됐다. Jazz Transformer의 REMI 스킴을 Music Transformer 위에서 구현했을 때, 피처 엔지니어링과 end-to-end 학습 사이의 관계가 체감됐다. 그리고 남의 표현 설계를 다른 아키텍처에 옮겨보는 것만으로도, 원 논문 두 편 중 어느 쪽도 답하지 않은 질문(이 표현이 recurrence 없는 relative attention과도 잘 맞는가)이 생긴다는 것도 배웠다.

또한, 비교를 설계하는 것과 비교를 완성하는 것은 다른 일이며, 측정할 수 없으면 주장할 수 없다는 것을 알게 되었다. 이 프로젝트의 다음 단계가 새 모델이 아니라 평가 지표인 이유다.

"Jazz is a Language"라는 비유가 단순한 은유 이상임을 구현을 통해 확인했다. 언어처럼, 재즈에도 어휘와 문법과 문맥이 있다. 그리고 언어 모델처럼, 무엇을 어휘로 삼을지 정하는 순간 모델이 배울 수 있는 것과 없는 것이 결정된다.

---

*시리즈 끝. 전체 코드: [Jazz-Is-a-Language](https://github.com/Jaeho-Jung/Jazz-Is-a-Language) · [MusicTransformer-Pytorch](https://github.com/Jaeho-Jung/MusicTransformer-Pytorch)*

*참고문헌: Huang, Y.-S., & Yang, Y.-H. (2020). Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions. ACM Multimedia. [arXiv:2002.00212](https://arxiv.org/abs/2002.00212) · Wu, S.-L., & Yang, Y.-H. (2020). The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures. ISMIR. [arXiv:2008.01307](https://arxiv.org/abs/2008.01307) · Huang, C.-Z. A., et al. (2018). Music Transformer: Generating Music with Long-Term Structure. ICLR. [arXiv:1809.04281](https://arxiv.org/abs/1809.04281)*
