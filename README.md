# function-calling

      
### prepare train data
```
python data/genererate_data.py
python data/preprocess_data.py
```

### lora finetuning
```
python train/main.py
```

### create an Excel file for qualitative evaluation
```
python test/lora_merge.py
python test/test.py
```
