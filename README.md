# MissGNN
## Usage
### Run the whole model

For regression tasks:
```
python train_mdi.py --pre_train uci --data [DATASET]
```
For classification tasks:
```
python train_mdi.py --task "class" --pre_train uci --data [DATASET]
```
### Run only the prediction model

For regression tasks:
```
python train_mdi.py uci --data [DATASET]
```
For classification tasks:
```
python train_mdi.py --task "class" uci --data [DATASET]
```
Add ``` --pre_train ``` if you want to apply pre-trained weights.
