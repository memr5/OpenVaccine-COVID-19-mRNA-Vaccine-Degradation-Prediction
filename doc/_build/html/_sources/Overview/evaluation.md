# Evaluation

Submissions are scored using MCRMSE (mean columnwise root mean squared error):

```{figure} Images/MCRMSE.png
---
name: MCRMSE
---
MCRMSE
```

where Nt is the number of scored ground truth target columns, and y and y^ are the actual and predicted values, respectively.

From the Data page: There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, degMgpH10, and degMg50C.



### Submission File

For each sample id in the test set, you must predict targets for each sequence position (seqpos), one per row. If the length of the sequence of an id is, e.g., 107, then you should make 107 predictions. Positions greater than the seq_scored value of a sample are not scored, but still need a value in the solution file.

```
    id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C    
    id_00073f8be_0,0.1,0.3,0.2,0.5,0.4
    id_00073f8be_1,0.3,0.2,0.5,0.4,0.2
    id_00073f8be_2,0.5,0.4,0.2,0.1,0.2
    etc.
```