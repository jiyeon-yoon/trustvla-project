# RunPod 처음부터 실행하기

이 문서는 TrustVLA Guard를 RunPod에서 돌리기 위한 순서입니다.
목표는 매번 LIBERO/OpenVLA dependency를 손으로 설치하지 않고,
GitHub Actions에서 만든 Docker image를 RunPod에서 바로 실행하는 것입니다.

## 0. 전체 구조

사용할 구조:

```text
GitHub repository
  -> GitHub Actions가 Docker image 빌드
  -> GHCR에 image 저장
  -> RunPod Pod 생성 시 container image로 사용
  -> /workspace에서 실험 실행
```

중요한 점:

```text
Docker image: dependency와 프로젝트 기본 코드
GitHub: 최신 소스코드
/workspace: RunPod에서 실행되는 작업 공간
Network Volume 또는 외부 백업: 큰 데이터, 모델 캐시, 실험 결과 보관
```

## 1. GitHub에 코드 push

로컬 Mac에서 먼저 변경사항을 push합니다.

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard

git status
git add README.md README_kor.md docs/runpod_setup_ko.md docs/runpod_docker_image.md docker/verify_runtime.py
git commit -m "Add Korean docs and RunPod setup guide"
git push
```

이미 commit할 변경이 없다면 `git status`에서 clean이라고 나옵니다.

## 2. GitHub Actions에서 Docker image 만들기

GitHub 웹사이트에서:

```text
jiyeon-yoon/trustvla-project
-> Actions
-> Build RunPod Image
-> Run workflow
```

tag 입력칸에는 새 버전을 넣습니다.

예:

```text
v0.4
```

빌드가 성공하면 사용할 image는 다음입니다.

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.4
```

주의:

```text
v0.4가 성공하기 전까지 v0.1 같은 이전 성공 image는 지우지 않는 것이 좋습니다.
새 image가 실패하면 이전 image가 fallback입니다.
```

## 3. image build가 확인하는 것

Docker image build 단계에서 확인하는 것:

```text
1. TrustVLA package 설치
2. pytest 통과
3. trustvla.cli doctor 실행
4. torch / transformers / PIL / numpy import 확인
5. LIBERO package 설치 여부와 basic import 확인
```

단, LIBERO full simulator는 MuJoCo/graphics/runtime 환경 영향을 받습니다.
그래서 최종 확인은 RunPod에서 Pod를 켠 뒤 다시 합니다.

```bash
source /workspace/activate_trustvla.sh
PYTHONPATH=src python -m trustvla.cli doctor
PYTHONPATH=src python -m pytest -q
```

## 4. RunPod에서 Pod 생성

RunPod에서 새 Pod를 만듭니다.

권장 시작 설정:

```text
GPU: RTX 4090
Container image: ghcr.io/jiyeon-yoon/trustvla-runpod:v0.4
Container disk: 최소 20GB, 여유 있으면 50GB
Volume disk: smoke test만 할 거면 20GB
HTTP port: 8888
TCP port: 22
```

Network Volume이 선택되지 않아도 당장 smoke test는 가능합니다.
다만 Network Volume이 없으면 Pod를 terminate할 때 `/workspace` 내용이 삭제됩니다.

반복 실험을 할 때는 다음이 좋습니다.

```text
Docker image: dependency 보관
GitHub: 코드 보관
Network Volume 또는 외부 저장소: LIBERO dataset, HF model cache, runs 결과 보관
```

## 5. RunPod에서 custom image 입력 위치

RunPod 화면에서 template/custom template을 만들 때 `Container Image` 또는
`Image Name` 입력칸에 아래를 넣습니다.

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.4
```

만약 image pull이 실패하면서 authorization 오류가 나면 GHCR package가 private일 수 있습니다.
그 경우 둘 중 하나가 필요합니다.

```text
1. GitHub Packages에서 ghcr.io/jiyeon-yoon/trustvla-runpod package를 public으로 바꾸기
2. RunPod template에 registry credential 설정하기
```

처음에는 public package가 가장 단순합니다.

## 6. Pod가 켜진 뒤 Web Terminal에서 확인

RunPod Pod 상세 화면에서:

```text
Connect
-> Web Terminal enable
-> Open web terminal
```

터미널에서:

```bash
ls /workspace
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

python --version
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m trustvla.cli doctor
```

기대 결과:

```text
pytest: passed
torch: ok
transformers: ok
PIL: ok
numpy: ok
```

`/workspace/activate_trustvla.sh`가 없으면 custom image로 뜬 것이 아닐 가능성이 큽니다.
그 경우 Pod template의 container image가 맞는지 다시 확인합니다.

## 7. SSH로 접속하기

RunPod Pod 상세 화면의 `Connect` 탭에서 `SSH over exposed TCP`를 봅니다.
예시는 다음처럼 생겼습니다.

```bash
ssh root@47.47.180.44 -p 14067 -i ~/.ssh/id_ed25519
```

실제 IP와 port는 Pod를 새로 만들 때마다 달라질 수 있습니다.
RunPod 화면에 보이는 값을 그대로 씁니다.

Mac 터미널에서:

```bash
ssh root@<RUNPOD_IP> -p <RUNPOD_PORT> -i ~/.ssh/id_ed25519
```

처음 접속할 때 아래 질문이 나오면:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

입력:

```text
yes
```

비밀번호를 물어보면 보통 SSH key가 적용되지 않은 것입니다.
RunPod template에 Mac의 public key가 들어갔는지 확인해야 합니다.

Mac public key 확인:

```bash
cat ~/.ssh/id_ed25519.pub
```

## 8. VSCode Remote SSH 연결

VSCode에서 `Remote - SSH` extension을 설치합니다.

그 다음 Mac의 SSH config를 엽니다.

```bash
code ~/.ssh/config
```

아래 block을 추가합니다.

```sshconfig
Host runpod-trustvla
    HostName <RUNPOD_IP>
    User root
    Port <RUNPOD_PORT>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
```

예:

```sshconfig
Host runpod-trustvla
    HostName 47.47.180.44
    User root
    Port 14067
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
```

VSCode에서:

```text
Command Palette
-> Remote-SSH: Connect to Host...
-> runpod-trustvla
```

중요:

```text
raw IP만 선택하지 말고 runpod-trustvla host를 선택해야 합니다.
그래야 port와 key 설정이 같이 적용됩니다.
```

접속 후 folder는 다음을 엽니다.

```text
/workspace/trustvla-project
```

## 9. 첫 smoke test 실행

RunPod에서:

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m trustvla.cli doctor
```

instruction benchmark 생성:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/seed_tasks.json \
  --out runs/smoke/generated_benchmark.jsonl
```

dummy rollout:

```bash
PYTHONPATH=src python -m trustvla.cli dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/dummy_rollouts.jsonl
```

guarded dummy rollout:

```bash
PYTHONPATH=src python -m trustvla.cli guard-dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/guarded_dummy_rollouts.jsonl
```

비교 리포트:

```bash
PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --rollout baseline=runs/smoke/dummy_rollouts.jsonl \
  --rollout guarded=runs/smoke/guarded_dummy_rollouts.jsonl \
  --out runs/smoke/comparison_report.md
```

확인할 파일:

```text
runs/smoke/comparison_report.md
```

## 10. LIBERO 데이터 다운로드

처음에는 작은 suite만 받습니다.

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

dry run만 보고 싶으면:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets \
  --dry-run
```

## 11. LIBERO seed task export

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 5 \
  --out data/libero_object_seed_draft.json
```

처음에는 `--limit 5`로만 확인합니다.
성공하면 나중에 `--limit 30`으로 늘립니다.

생성된 파일에서 다음 field는 사람이 확인/보강해야 할 수 있습니다.

```text
target_object
possible_objects
distractor_objects
absent_objects
ambiguous_targets
safety_hazards
```

## 12. Paired instruction benchmark 생성

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

## 13. OpenVLA raw rollout

처음에는 작은 값으로 실행합니다.

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --trace-dir runs/libero_object/traces/openvla
```

## 14. OpenVLA + Guard rollout

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --guarded \
  --trace-dir runs/libero_object/traces/openvla_guarded
```

## 15. Detector와 compare

raw rollout detector:

```bash
PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.jsonl \
  --out runs/libero_object/openvla_rollouts.detected.jsonl
```

guarded rollout detector:

```bash
PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_guarded_rollouts.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.detected.jsonl
```

비교 리포트:

```bash
PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollout baseline=runs/libero_object/openvla_rollouts.detected.jsonl \
  --rollout guarded=runs/libero_object/openvla_guarded_rollouts.detected.jsonl \
  --out runs/libero_object/openvla_comparison_report.md
```

논문 table 후보:

```text
runs/libero_object/openvla_comparison_report.md
```

## 16. Pod 종료 전 해야 할 일

Terminate하면 `/workspace`가 사라질 수 있습니다.
종료 전 아래를 확인합니다.

```bash
cd /workspace/trustvla-project
git status
```

코드 변경이 있으면:

```bash
git add .
git commit -m "Add RunPod experiment results"
git push
```

결과 파일은 너무 크면 GitHub에 바로 올리지 말고 따로 백업합니다.

중요 결과:

```text
runs/libero_object/openvla_rollouts.jsonl
runs/libero_object/openvla_guarded_rollouts.jsonl
runs/libero_object/openvla_comparison_report.md
runs/libero_object/traces/
```

비용을 아끼려면 작업이 끝난 뒤 Pod는 stop이 아니라 terminate합니다.
단, terminate 전에 결과 백업이 끝나 있어야 합니다.

## 17. 자주 나는 문제

### activate_trustvla.sh가 없다

```text
원인: custom image로 뜬 것이 아니거나 start script가 실행되지 않음
확인: RunPod template의 container image가 ghcr.io/jiyeon-yoon/trustvla-runpod:v0.4인지 확인
```

### VSCode가 raw IP로 접속 실패

```text
원인: port/key 설정이 빠짐
해결: ~/.ssh/config에 runpod-trustvla host를 만들고 그 host로 접속
```

### 비밀번호를 물어본다

```text
원인: SSH public key가 Pod에 등록되지 않음
해결: RunPod SSH key 설정 확인
```

### image pull authorization 오류

```text
원인: GHCR package가 private일 가능성
해결: package public 전환 또는 RunPod registry credential 설정
```

### LIBERO가 doctor에서 missing

```text
원인: 잘못된 image이거나 Docker build에서 LIBERO 설치가 빠짐
해결: image tag 확인, GitHub Actions build log 확인
```

### detector 결과에서 selected_target이 비어 있음

```text
원인: LIBERO rollout trace의 info key와 detector가 기대하는 key가 다름
해결: runs/libero_object/traces/*.json을 보고 detectors.py를 실제 key에 맞게 조정
```
