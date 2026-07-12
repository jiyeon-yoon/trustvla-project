# RunPod 세팅과 첫 실험 실행

이 파일 하나만 보고 RunPod를 켜고, TrustVLA의 첫 실제 LIBERO/OpenVLA 실험까지
진행하면 됩니다.

지금은 헷갈리지 않도록 Pod 생성, SSH/VSCode 접속, image 확인, 실제 rollout 실행을
이 파일 하나에 합쳤습니다. `docs/` 안의 RunPod 문서는 이제 이 파일 하나만 남겼습니다.

## 1. 전체 구조

우리가 쓰는 방식은 다음입니다.

```text
GitHub repository
  -> GitHub Actions가 Docker image 빌드
  -> GHCR에 image 저장
  -> RunPod Pod 생성 시 container image로 사용
  -> /workspace에서 실험 실행
```

역할:

```text
Docker image: 설치 오래 걸리는 dependency 보관
GitHub: 최신 소스코드 보관
/workspace: RunPod에서 실행되는 작업 공간
Network Volume 또는 외부 백업: 모델 캐시, 데이터셋, 결과 보관
```

## 2. 로컬 Mac에서 코드 push

RunPod image를 만들기 전에 로컬 변경사항을 GitHub에 올립니다.

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard

git status
git add .
git commit -m "Update TrustVLA RunPod workflow"
git push
```

이미 commit할 것이 없으면 `git status`가 clean이라고 나옵니다.

## 3. GitHub Actions에서 Docker image 만들기

GitHub 웹사이트에서:

```text
jiyeon-yoon/trustvla-project
-> Actions
-> Build RunPod Image
-> Run workflow
```

tag 입력칸에는 새 버전을 넣습니다.

```text
v0.7
```

빌드가 성공하면 RunPod에서 사용할 image는 다음입니다.

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.7
```

주의:

```text
새 버전이 성공하기 전까지 이전 성공 image는 지우지 않습니다.
새 image가 실패하면 이전 image가 fallback입니다.
```

## 4. image build가 확인하는 것

Docker image build 단계에서 확인하는 것:

```text
1. TrustVLA package 설치
2. pytest 통과
3. trustvla.cli doctor 실행
4. torch / transformers / PIL / numpy import 확인
5. LIBERO package basic import 확인
```

단, LIBERO full simulator는 MuJoCo/graphics/runtime 환경 영향을 받습니다. 그래서
RunPod Pod를 실제로 켠 뒤 다시 확인해야 합니다.

## 4-1. Docker image에 들어있는 것과 없는 것

`ghcr.io/jiyeon-yoon/trustvla-runpod:v0.7` image에 들어있는 것:

```text
Python 3.10 virtual environment: /opt/trustvla-env
PyTorch / torchvision CUDA wheels
OpenVLA Hugging Face runtime dependencies
LIBERO code: /opt/LIBERO
LIBERO Python package editable install
TrustVLA code copy: /opt/trustvla-project
RunPod startup script: /start-trustvla.sh
```

image에 들어있지 않은 것:

```text
OpenVLA 7B model weights
LIBERO datasets
실험 결과
논문용 checkpoint
```

큰 파일은 `/workspace`, Network Volume, Hugging Face cache, 또는 외부 저장소에 둡니다.
OpenVLA weight는 기본적으로 아래에 cache됩니다.

```text
/workspace/.cache/huggingface
```

Docker Desktop이 있는 로컬 Mac에서 직접 build할 수도 있지만, 지금 기본 흐름은
GitHub Actions로 GHCR image를 만드는 것입니다.

## 5. RunPod custom template 만들기

RunPod에서 먼저 custom template을 만듭니다.

```text
Templates
-> New Template 또는 Create Template
```

입력:

```text
Template name: trustvla-v0.7
Container Image: ghcr.io/jiyeon-yoon/trustvla-runpod:v0.7
```

Port:

```text
HTTP Port: 8888
TCP Port: 22
```

Start command 또는 Docker command 칸은 비워둬도 됩니다. 꼭 입력해야 하면:

```bash
/start-trustvla.sh
```

만약 image pull authorization 오류가 나면 GHCR package가 private일 수 있습니다.
처음에는 GitHub Packages에서 `ghcr.io/jiyeon-yoon/trustvla-runpod` package를 public으로
바꾸는 것이 가장 단순합니다.

## 6. 새 Pod 생성

RunPod에서 새 Pod를 만들 때:

```text
GPU: RTX 4090
Template: trustvla-v0.7
Container disk: 50GB 권장
Volume disk: 20GB 이상
HTTP port: 8888
TCP port: 22
```

Network Volume이 없어도 첫 smoke test는 가능합니다. 다만 Network Volume이 없으면
Pod를 terminate할 때 `/workspace` 내용이 삭제됩니다.

반복 실험에 가장 좋은 구조:

```text
Docker image: dependency 보관
GitHub: 코드 보관
Network Volume 또는 외부 저장소: LIBERO dataset, HF model cache, runs 결과 보관
```

## 7. Pod가 제대로 뜬 것인지 확인

RunPod에서:

```text
Connect
-> Web Terminal enable
-> Open web terminal
```

터미널에서:

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

python -m pytest -q
python -m trustvla.cli doctor
nvidia-smi
```

기대 결과:

```text
pytest passed
libero: ok
torch: ok
transformers: ok
PIL: ok
numpy: ok
GPU visible in nvidia-smi
```

`/workspace/activate_trustvla.sh`가 없으면 custom image로 뜬 것이 아닙니다.
Pod template의 `Container Image`가 맞는지 다시 확인합니다.

만약 `/opt/LIBERO`와 `pip show libero`는 보이는데 `doctor`에서 `libero: missing`이
나오면, 대부분 `PYTHONPATH=src python ...`처럼 실행해서 activate script의
`PYTHONPATH`를 덮어쓴 것입니다. 현재 Pod에서는 아래처럼 바로 고칩니다.

```bash
sed -i 's|export PYTHONPATH=.*|export PYTHONPATH=/workspace/trustvla-project/src:/opt/LIBERO|' /workspace/activate_trustvla.sh
source /workspace/activate_trustvla.sh

python -m trustvla.cli doctor
python - <<'PY'
import libero
print(libero.__file__)
PY
```

RunPod에서는 `source /workspace/activate_trustvla.sh` 이후 `PYTHONPATH=src`를 앞에
붙이지 않습니다.

## 8. SSH로 접속하기

RunPod Pod 상세 화면의 `Connect` 탭에서 `SSH over exposed TCP` 명령어를 봅니다.
예시는 다음처럼 생겼습니다.

```bash
ssh root@47.47.180.44 -p 14067 -i ~/.ssh/id_ed25519
```

실제 IP와 port는 Pod를 새로 만들 때마다 달라질 수 있습니다. RunPod 화면에 보이는
값을 그대로 씁니다.

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

비밀번호를 물어보면 보통 SSH key가 적용되지 않은 것입니다. RunPod SSH key 설정에
Mac public key가 들어갔는지 확인합니다.

Mac public key 확인:

```bash
cat ~/.ssh/id_ed25519.pub
```

## 9. VSCode Remote SSH 연결

Mac에서 SSH config를 엽니다.

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
raw IP를 선택하지 말고 runpod-trustvla host를 선택합니다.
그래야 port와 key 설정이 같이 적용됩니다.
```

접속 후 folder는 다음을 엽니다.

```text
/workspace/trustvla-project
```

## 10. 첫 실험 크기

처음부터 큰 실험을 돌리지 않습니다. 첫 RunPod 세션 목표는 아래입니다.

```text
LIBERO task: 1개
init state: 1개
rollout 종류: raw OpenVLA 먼저 1개
max steps: 50
```

첫 성공 기준:

```text
runs/pilot/openvla_raw.jsonl
runs/pilot/traces/raw/<case_id>.json
```

이 두 파일이 생기고 trace 안에 action/reward/contact 정보가 들어오면 1차 성공입니다.

## 10-1. LIBERO 데이터와 Hugging Face 다운로드

전체 LIBERO dataset mirror를 받을 필요는 없습니다. 첫 pilot은 `libero_object`만 씁니다.

Hugging Face mirror:

```text
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```

첫 다운로드는 dry run부터 봅니다.

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets \
  --dry-run
```

괜찮으면 실제 다운로드:

```bash
python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

LIBERO 공식 script를 써야 하는 환경이면:

```bash
cd /opt/LIBERO
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --use-huggingface
```

처음부터 `all`을 받지 않습니다. pipeline이 돈 뒤에 `libero_spatial`을 추가합니다.

## 11. LIBERO seed 1개 export

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project

python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 1 \
  --out data/libero_object_seed_draft.json
```

생성된 파일 확인:

```bash
python -m json.tool data/libero_object_seed_draft.json | head -120
```

수동으로 확인해야 하는 필드:

```text
target_object
possible_objects
distractor_objects
absent_objects
ambiguous_targets
safety_hazards
metadata.libero_task_id
```

자동 export가 일부 필드를 채워도 그대로 믿지 않습니다. 특히 `safety_hazards`는 논문의
trusted safety policy가 될 수 있으므로 사람이 확인해야 합니다.

## 12. Safety policy draft 생성

```bash
python -m trustvla.cli export-safety-policies \
  --seed-tasks data/libero_object_seed_draft.json \
  --out data/libero_object_safety_policies_draft.json
```

확인:

```bash
python -m json.tool data/libero_object_safety_policies_draft.json | head -160
```

여기서 protected object가 실제 장면의 hazard와 맞는지 봅니다. 이 파일은 benchmark
정답이 아니라, runtime gate가 읽는 별도 safety policy입니다.

## 13. Benchmark 생성과 검증

```bash
mkdir -p runs/pilot

python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --init-states 1 \
  --out runs/pilot/benchmark.jsonl
```

검증:

```bash
python -m trustvla.cli validate-benchmark \
  --benchmark runs/pilot/benchmark.jsonl \
  --safety-policies data/libero_object_safety_policies_draft.json
```

`native_success_not_valid` warning은 target/spatial edit에서 나올 수 있습니다. 이 warning은
"LIBERO 원래 reward를 그대로 counterfactual success로 쓰면 안 된다"는 뜻입니다. 첫 raw
rollout 디버깅 단계에서는 괜찮지만, 논문 결과에서는 별도 evaluator가 필요합니다.

## 14. Raw OpenVLA rollout 1개 실행

처음에는 guard와 grounding prompt를 끄고 raw OpenVLA만 돌립니다.

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/pilot/benchmark.jsonl \
  --out runs/pilot/openvla_raw.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --trace-dir runs/pilot/traces/raw
```

중간에 끊겼으면 같은 명령 끝에 `--resume`을 붙입니다.

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/pilot/benchmark.jsonl \
  --out runs/pilot/openvla_raw.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --trace-dir runs/pilot/traces/raw \
  --resume
```

## 15. 결과 파일 확인

```bash
ls -lh runs/pilot
ls -lh runs/pilot/traces/raw
head -1 runs/pilot/openvla_raw.jsonl
```

trace JSON 하나를 확인합니다.

```bash
python -m json.tool runs/pilot/traces/raw/*.json | head -200
```

확인할 것:

```text
case_id가 benchmark case와 연결되는가
actions가 기록되는가
reward 또는 success 정보가 들어오는가
native_success가 기록되는가
instruction_success가 기록되는가
trustvla_contacts 또는 contact 관련 정보가 기록되는가
```

여기까지 성공하면 RunPod 세션 1차 목표는 달성입니다.

## 16. Language emphasis 실행

raw가 성공한 뒤에만 실행합니다.

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/pilot/benchmark.jsonl \
  --out runs/pilot/openvla_language_emphasis.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --grounding-mode language_emphasis \
  --trace-dir runs/pilot/traces/language_emphasis
```

`language_emphasis`는 cheap pilot입니다. 논문 본실험에서는 CAG/IGAR 같은 재현 가능한
grounding baseline을 추가해야 합니다.

## 17. Safety gate 실행

language emphasis가 성공한 뒤 실행합니다.

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/pilot/benchmark.jsonl \
  --out runs/pilot/openvla_gated.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --grounding-mode language_emphasis \
  --guarded \
  --safety-policies data/libero_object_safety_policies_draft.json \
  --trace-dir runs/pilot/traces/gated
```

이 gate는 benchmark의 `expected_behavior`나 `safety_class`를 보지 않습니다. raw
instruction과 별도 safety policy만 보고 차단 여부를 결정합니다.

## 18. 점수 계산

raw rollout:

```bash
python -m trustvla.cli tradeoff-score \
  --benchmark runs/pilot/benchmark.jsonl \
  --rollouts runs/pilot/openvla_raw.jsonl

python -m trustvla.cli pair-score \
  --benchmark runs/pilot/benchmark.jsonl \
  --rollouts runs/pilot/openvla_raw.jsonl \
  --difference-threshold 0.05 \
  --prefix-steps 10
```

세 조건 비교 report:

```bash
python -m trustvla.cli compare \
  --benchmark runs/pilot/benchmark.jsonl \
  --rollout raw=runs/pilot/openvla_raw.jsonl \
  --rollout language=runs/pilot/openvla_language_emphasis.jsonl \
  --rollout gated=runs/pilot/openvla_gated.jsonl \
  --out runs/pilot/comparison_report.md
```

## 19. Jupyter Notebook으로 실행

RunPod Jupyter에서 아래 파일을 열 수 있습니다.

```text
/workspace/trustvla-project/notebooks/runpod_libero_openvla.ipynb
```

처음 값은 다음처럼 작게 둡니다.

```python
LIMIT = 1
INIT_STATES = 1
MAX_STEPS = 50
RUN_ROLLOUTS = True
```

노트북은 편하지만, 에러가 났을 때는 terminal 명령어가 더 추적하기 쉽습니다.

## 20. Pod 종료 전 체크리스트

Network Volume 없이 Pod를 terminate하면 `/workspace` 내용이 사라집니다.

종료 전 확인:

```bash
cd /workspace/trustvla-project
ls -lh runs/pilot
git status
```

보관해야 하는 파일:

```text
runs/pilot/*.jsonl
runs/pilot/**/*.json
runs/pilot/*.md
data/libero_object_seed_draft.json
data/libero_object_safety_policies_draft.json
```

코드 변경이 있으면 GitHub에 push하고, 결과 파일은 로컬로 가져오거나 별도 저장소에
백업합니다. 비용을 아끼려면 작업이 끝난 뒤 Pod는 stop이 아니라 terminate합니다.
단, terminate 전에 결과 백업이 끝나 있어야 합니다.

## 21. 다음 확장 기준

아래가 모두 확인되면 실험 크기를 키웁니다.

```text
1. doctor에서 libero/torch/transformers가 ok
2. raw OpenVLA 1개 rollout 성공
3. trace JSON에 action/reward/contact 정보가 있음
4. image key와 action unnormalization 문제가 없음
5. gated condition이 benchmark label 없이 동작함
```

그 다음 확장 순서:

```text
1 task x 1 init
5 task x 1 init
5 task x 3 init
20-30 task x 3 init
두 번째 VLA 또는 grounding baseline 추가
```

## 22. 자주 나는 문제

### activate_trustvla.sh가 없다

```text
원인: custom image로 뜬 것이 아니거나 start script가 실행되지 않음
확인: RunPod template의 container image가 ghcr.io/jiyeon-yoon/trustvla-runpod:v0.7인지 확인
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
해결: runs/pilot/traces/*.json을 보고 detectors.py를 실제 key에 맞게 조정
```
