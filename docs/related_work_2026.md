# 2026년 VLA Safety/Language 선행연구와 TrustVLA의 위치

검색 기준일은 2026-07-12입니다. 범위는 VLA instruction robustness, language
grounding, paired/counterfactual evaluation, unsafe success, ambiguity/abstention,
runtime guard입니다. 2026년 논문 다수는 peer-reviewed 확정 논문이 아니라 arXiv
preprint이지만, 신규성 판단과 reviewer 비교 대상에는 포함해야 합니다.

## 직접 겹치는 연구

| 연구 | 주요 내용 | 이전 TrustVLA 주장과 겹친 부분 |
|---|---|---|
| [LIBERO-Plus](https://arxiv.org/abs/2510.13626) | 7개 perturbation 축, language ignorance | language perturbation 평가 |
| [LIBERO-CF](https://arxiv.org/abs/2602.17659) | 같은 시각 장면의 counterfactual instruction, CAG | paired instruction과 train-free 완화 |
| [Metamorphic Testing of VLA-Enabled Robots](https://arxiv.org/abs/2602.22579) | 입력 변형 전후 trajectory metamorphic relation | action/trajectory consistency 평가 |
| [ICBench + IGAR](https://arxiv.org/abs/2603.06001) | 같은 LIBERO 장면의 모순/OOD instruction | impossible instruction과 train-free guard |
| [LIBERO-Para](https://arxiv.org/abs/2603.28301) | paraphrase robustness와 trajectory divergence | paraphrase 변형과 metric |
| [ProGAL-VLA](https://arxiv.org/abs/2604.09824) | verified grounding, ambiguity detection, clarification | ambiguity/clarification |
| [STRONG-VLA](https://arxiv.org/abs/2604.10055) | 28개 multimodal perturbation | language/visual robustness |
| [HazardArena](https://arxiv.org/abs/2604.12447) | safe/unsafe twin scene, training-free Safety Option Layer | paired safety와 inference-time gate |
| [RedVLA](https://arxiv.org/abs/2604.22591) | physical red teaming, SimpleVLA-Guard | lightweight safety guard |
| [RoVLA](https://arxiv.org/abs/2605.19678) | instruction/trajectory/observation consistency | action consistency |
| [SafeVLA-Bench](https://arxiv.org/abs/2606.00773) | success-safety gap, SBU/VSI, STL | unsafe success |
| [RoboSemanticBench](https://arxiv.org/abs/2606.02277) | semantics가 physical target 선택에 반영되는지 진단 | language-action grounding |
| [LIBERO-Safety](https://arxiv.org/abs/2606.23686) | 대규모 physical/semantic safety benchmark | LIBERO safety benchmark |
| [ForesightSafety-VLA](https://arxiv.org/abs/2606.27079) | Safe-Lang/Safe-Vis/Safe-Core와 unsafe success | instruction-side safety |
| [SafeVLA](https://arxiv.org/abs/2503.03480) | CMDP와 constrained learning 기반 safety alignment | safety-performance trade-off |

분야 전체 구조는 [Vision-Language-Action Safety Survey](https://arxiv.org/abs/2604.23775)를
출발점으로 사용합니다.

## 왜 이전 주장은 충분하지 않았는가

이전 주장의 구성요소를 따로 보면 이미 각각 강한 선행연구가 있습니다.

```text
paired instruction       -> LIBERO-CF, ICBench, Metamorphic Testing
paraphrase               -> LIBERO-Para
trajectory consistency   -> Metamorphic Testing, RoVLA
unsafe success           -> SafeVLA-Bench, ForesightSafety-VLA
ambiguity/clarification  -> ProGAL-VLA
train-free guard         -> HazardArena, RedVLA, IGAR, CAG
```

따라서 이들을 단순히 한 benchmark에 합치는 것만으로는 탑티어 신규성이 부족합니다.

## 변경한 연구 질문

TrustVLA는 language capability와 safety를 각각 측정하는 데서 끝내지 않고 둘의
상호작용을 연구합니다.

> Language-grounding intervention이 benign instruction compliance를 높일 때,
> trusted safety policy와 충돌하는 hazardous instruction compliance도 함께
> 높이는가?

이 질문을 위해 같은 장면에서 safe executable instruction과 policy-conflicting
instruction을 matched pair로 만들고 다음 두 축을 동시에 보고합니다.

```text
x축: benign instruction compliance
y축: hazardous instruction compliance (낮을수록 좋음)
```

좋은 VLA는 x축이 높고 y축이 낮아야 합니다. 두 축을 하나의 평균 success로 합치면
언어를 무시해서 우연히 안전한 모델과, 언어를 잘 이해하면서 위험 지시를 거부하는
모델을 구분할 수 없습니다.

## 아직 검증해야 할 신규성 위험

현재 검색에서는 이 safety-obedience interaction을 중심 주장으로 삼은 직접 동일
논문을 찾지 못했습니다. 그렇다고 신규성이 확정된 것은 아닙니다. 제출 전에는 각
선행연구 본문의 experiment matrix와 최신 citation을 다시 확인해야 합니다.

특히 HazardArena와 ForesightSafety-VLA가 가장 가까운 비교 대상입니다. 논문에서는
다음 차이를 실험으로 입증해야 합니다.

1. 단순 safety benchmark가 아니라 grounding 강화 전후의 **위험 복종 증가량**을 측정한다.
2. `harmful compliance`와 `over-refusal`을 함께 보고 selective behavior를 평가한다.
3. guard는 benchmark answer label이 아니라 독립된 trusted policy만 사용한다.
4. 같은 model/scene/init state에서 grounding condition만 바꾼 paired statistics를 사용한다.
