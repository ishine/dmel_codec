# steps

1. complete config/person/yours.yaml
2. run generate_filelist.py
```
python generate_filelist.py person=zh_magia_v100
```
3. run experiment
```
python train.py person=zh_magia_v100 experiment=dmel_pretrain
python train.py person=zh_magia_v100 experiment=fsq_pretrain
python train.py person=zh_magia_v100 experiment=dmel_pertrain_ngroups10_ncodebooks1_levels86_downsample22_encoderresiduallayers16
```