## 환경설정 (돌려서 모듈 모자라다거나 에러뜨면 GPT한테 도움요청, 맘바가 리눅스에서만 돌아가요. widows 안됨)

```sh
conda create -n aaa python=3.10
conda activate aaa
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
configs/Market/a.yml에서 PRETRAIN_PATH, DATASETS ROOT_DIR, OUTPUT_DIR 본인위치로 바꿔주세요.

## 데이터셋(market1501) & Pretrain Model(vmamba base)
```sh
https://drive.google.com/drive/folders/1wU5IC4HBZd0ut1KqvEs17XDyCoXZ44ei?usp=sharing
```
폴더에 archive.zip이 마켓1501 데이터셋이고 pth로 끝나는 파일이 pretrain모델입니다.<br>
아직 설정이 market1501만 잘 돼있어서 다른 데이터셋 쓸려면 알아서 yml파일 만들어야 돼요.<br>
market의 a.yml복붙한담에 DATASETS: NAMES: ROOT_DIR: 이 부분 바꾸면 됩니다.

## Training
600 epoch 해야 잘나옴<br>
training command:

```bash
python train.py --config_file configs/Market/a.yml
```

## Evaluation

evaluation command:

```bash
python test1.py --config_file configs/Market/a.yml TEST.WEIGHT '모델저장위치/vssm_에폭(모델이름).pth'
```
test파일은 그냥 측정이고 test1파일은 flops,throghput,latency가 뜸

## 보통 고치는 파일
Vmamba백본인데 레이어 머징 기능 추가함:/model/backbones/vmamba.py<br>
위에서 Vmamba관련 코드 건드리면 고치는거(VSSM):/model/make_model.py<br>
훈련이랑 추론 설정:/processor/processor.py<br>
옵션설정(보통 추가만 하고 설정 고치는건 yml파일에 설정해놓음):/config/defaults.py<br>
옵션설정(주로 여기서 설정 고치고 최적화):/configs/Market/a.yml<br>

## 고칠때 순서
vmamba.py에서 중요도 점수라던가 적용 레이어라던가 그런거 바꾸는거 시도가능.<br>
위 파일의 VSSM 설정을 건드렸으면 make_model.py에서 고친 부분 추가.<br>
defaults.py는 웬만하면 안건드려도 되나 추가한 옵션이 있거나 옵션에 어떤게 있는지 참고할때 건드리기.<br>
a.yml에서 PRETRAIN_PATH, DATASETS, OUTPUT_DIR 같은 설정들을 본인 폴더에 맞게 고쳐주셈.<br>
깨끗한 vmamba 코드는 원본 Vmamba논문의 코드를 보세요.<br>
```bash
https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py
```
## SOTA
이 논문에서 보시면 됩니다.
```bash
https://arxiv.org/pdf/2511.07948
```

## 최적화 도와주세요
한번 돌리는데 6-7시간 걸려요 이것 뭐에요
