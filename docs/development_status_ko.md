# 개발 상태와 다음 마일스톤

기준일: 2026-07-12

## 고정한 연구 질문

> 언어 grounding을 강화하면 안전한 instruction뿐 아니라 trusted safety policy와
> 충돌하는 위험한 instruction compliance도 증가하는가? 외부 safety-policy gate가
> benign compliance를 유지하면서 hazardous compliance만 낮출 수 있는가?

## 구현 상태

| 영역 | 상태 | 비고 |
|---|---|---|
| safe/hazardous matched case 생성 | 완료 | `pair_id`, `safety_class`, multi-init 지원 |
| safety policy 분리 | 완료 | 별도 JSON, draft export, validator |
| 비-oracle semantic gate | 완료 | benchmark 정답 label을 decision input으로 사용하지 않음 |
| trade-off metric | 완료 | BIC, HIC, SCC, abstention, over-refusal, SOS |
| 실제 trajectory pair metric | 완료 | action trace prefix RMSE |
| LIBERO native success 분리 | 완료 | `native_success`, `instruction_success` |
| MuJoCo contact logging | 구현, 실환경 검증 필요 | simulator가 `sim`을 노출할 때 geom pair 저장 |
| OpenVLA adapter | 구현, 실환경 검증 필요 | 모델 load부터 rollout까지 GPU에서 확인 필요 |
| prompt grounding pilot | 완료 | `language_emphasis`; 논문 main method는 아님 |
| CAG/IGAR 등 선행 baseline | 미구현 | 논문 규모 실험 전에 필요 |
| 실제 LIBERO rollout | 0개 | 현재 가장 큰 blocker |
| 실제 논문 결과/통계 | 0% | synthetic 값은 사용 금지 |

주관적인 진행률:

- 로컬 연구 인프라: 약 60%
- GPU 실험 파이프라인 검증: 약 20%
- 논문 전체: 약 20-25%

## 다음 RunPod 세션의 성공 조건

처음부터 81개 case를 전부 실행하지 않습니다. 다음 순서를 지킵니다.

1. Docker image `v0.5`에서 `pytest`와 `doctor` 확인.
2. LIBERO task 1개를 export하고 annotation 확인.
3. `--init-states 1`, base case 1개, `--max-steps 50`으로 raw rollout.
4. trace JSON에 image/action/reward와 `trustvla_contacts`가 기록되는지 확인.
5. rollout 영상을 보거나 frame을 확인해 action unnormalization이 맞는지 검증.
6. 그 다음에만 raw/language-emphasis/gated 세 조건으로 확대.

첫 세션의 완료 증거:

```text
runs/pilot/openvla_raw.jsonl
runs/pilot/traces/raw/<case_id>.json
```

## 논문 실험 전에 반드시 남은 일

1. target/spatial edit의 counterfactual success predicate 구현.
2. 실제 MuJoCo geom name과 annotation object ID 매핑.
3. CAG/IGAR 또는 공개 코드가 있는 grounding 방법 최소 1개 연결.
4. 두 번째 VLA family 연결.
5. task와 initial state를 block으로 둔 paired bootstrap confidence interval.
6. safety policy 독립 annotation 및 inter-rater agreement.
7. 실패 사례 영상과 trajectory plot 생성.

## Go/No-Go

5-task pilot에서 grounding condition이 raw 대비 BIC 차이를 보이지 않거나, 모든 모델의
HIC가 0이면 현재 central claim을 유지하지 않습니다. 반대로 BIC와 HIC가 동시에
증가하고 gate가 HIC만 낮추면 20-30 task 실험으로 확장합니다.
