# Stratification by the signal_to_noise

* We binned samples on signal_to_noise ratio and used Stratified KFold on it.

```python
def get_stratify_group(row):
    snf = row['SN_filter']
    snr = row['signal_to_noise']
    
    if snf == 0:
        if snr<0:
            snr_c = 0
        elif 0<= snr < 2:
            snr_c = 1
        elif 2<= snr < 4:
            snr_c = 2
        elif 4<= snr < 5.5:
            snr_c = 3
        elif 5.5<= snr < 10:
            snr_c = 4
        elif snr >= 10:
            snr_c = 5
            
    else: # snf == 1
        if snr<0:
            snr_c = 6
        elif 0<= snr < 1:
            snr_c = 7
        elif 1<= snr < 2:
            snr_c = 8
        elif 2<= snr < 3:
            snr_c = 9
        elif 3<= snr < 4:
            snr_c = 10
        elif 4<= snr < 5:
            snr_c = 11
        elif 5<= snr < 6:
            snr_c = 12
        elif 6<= snr < 7:
            snr_c = 13
        elif 7<= snr < 8:
            snr_c = 14
        elif 8<= snr < 9:
            snr_c = 15
        elif 9<= snr < 10:
            snr_c = 16
        elif snr >= 10:
            snr_c = 17
        
    return '{}'.format(snr_c)

train['stratify_group'] = train.apply(get_stratify_group, axis=1)
train['stratify_group'] = train['stratify_group'].astype('category').cat.codes

skf = StratifiedKFold(n_folds, shuffle=True, random_state=53)
```

* Here are the distributions across all folds:

```{figure} Images/starified_snr.png
---
name: Stratification by the signal_to_noise
---
Stratification by the signal_to_noise
```

