# next-prediction.pytorch
 
This repo is the unofficial reimplementation of pytorch version of [next-prediction](https://github.com/google/next-prediction).

We strongly follow the structure of [next-prediction](https://github.com/google/next-prediction) and change each line to PyTorch to make sure the model pipeline is consistent, except that we add `pip install wandb` to monitor the process.

We have trained and tested the model. Here is the results:

```
performance:
0000_ade, 19.522013
0000_fde, 40.70279
0002_ade, 13.291343
0002_fde, 27.356066
0400_ade, 14.328231
0400_fde, 30.023241
0401_ade, 24.559025
0401_fde, 50.8526
0500_ade, 24.984905
0500_fde, 50.220814
act_ap, 0.19538545473261743
ade, 19.182003
fde, 39.686977
grid1_acc, 0.29150116822429906
grid2_acc, 0.3939544392523365
mov_ade, 21.836597
mov_fde, 45.75023
static_ade, 14.864564
static_fde, 29.825686
traj_class_accuracy, 0.9222352252779403
traj_class_accuracy_0, 0.8669125557092362
traj_class_accuracy_1, 0.9562505905697817
0000_ade 0000_fde 0002_ade 0002_fde 0400_ade 0400_fde 0401_ade 0401_fde 0500_ade 0500_fde act_ap ade fde grid1_acc grid2_acc mov_ade mov_fde static_ade static_fde traj_class_accuracy traj_class_accuracy_0 traj_class_accuracy_1
19.522013 40.70279 13.291343 27.356066 14.328231 30.023241 24.559025 50.8526 24.984905 50.220814 0.19538545473261743 19.182003 39.686977 0.29150116822429906 0.3939544392523365 21.836597 45.75023 14.864564 29.825686 0.9222352252779403 0.8669125557092362 0.9562505905697817
```
The raw code result is 

<table>
  <tr>
    <td>Activity mAP</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.199</td>
    <td>17.979</td>
    <td>37.176</td>
  </tr>
</table>

There are two possible bugs here we have not tested:
* the pytorch version of tf.embedding_lookup
* the pytorch version of tf.gather_nd


If you find this code useful in your research then please cite

```
@InProceedings{Liang_2019_CVPR,
  author = {Liang, Junwei and Jiang, Lu and Carlos Niebles, Juan and Hauptmann, Alexander G. and Fei-Fei, Li},
  title = {Peeking Into the Future: Predicting Future Person Activities and Locations in Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
