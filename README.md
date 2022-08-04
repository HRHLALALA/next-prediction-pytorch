# next-prediction.pytorch
 
This repo is the unofficial PyTorch reimplementation of [next-prediction](https://github.com/google/next-prediction).

We strongly follow the structure of [next-prediction](https://github.com/google/next-prediction) and change each line to PyTorch to make sure the model pipeline is consistent, except that we add `pip install wandb` to monitor the process.

We have trained and tested the model. Here is the results:

```
performance:
0000_ade, 19.19477
0000_fde, 39.99288
0002_ade, 13.106992
0002_fde, 26.834414
0400_ade, 13.9344225
0400_fde, 29.386696
0401_ade, 23.902508
0401_fde, 49.456684
0500_ade, 25.185291
0500_fde, 50.284653
act_ap, 0.19936162447332648
ade, 18.790426
fde, 38.814884
grid1_acc, 0.29246495327102806
grid2_acc, 0.3962908878504673
mov_ade, 21.366684
mov_fde, 44.637093
static_ade, 14.600393
static_fde, 29.345638
traj_class_accuracy, 0.9291691047396138
traj_class_accuracy_0, 0.8854310742277547
traj_class_accuracy_1, 0.9560616082396296
0000_ade 0000_fde 0002_ade 0002_fde 0400_ade 0400_fde 0401_ade 0401_fde 0500_ade 0500_fde act_ap ade fde grid1_acc grid2_acc mov_ade mov_fde static_ade static_fde traj_class_accuracy traj_class_accuracy_0 traj_class_accuracy_1
19.19477 39.99288 13.106992 26.834414 13.9344225 29.386696 23.902508 49.456684 25.185291 50.284653 0.19936162447332648 18.790426 38.814884 0.29246495327102806 0.3962908878504673 21.366684 44.637093 14.600393 29.345638 0.9291691047396138 0.8854310742277547 0.9560616082396296
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

## Train/Test
Please use the raw scripts in [next-prediction](https://github.com/google/next-prediction) to train the model or use `bash scripts/run.sh ${your args}`. 

Note on `scripts/run.sh`: 
* args need to be assigned using the format `--{key}={0/1}` for booleans or `--{key}={value}` for others. 
* Use `--run_mode=train/test_single` to select whether training or testing

Welcome to contribute to this reimplementation and let me know if you find bugs!

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

### Updates
* **29 July 2022**: We have added `kaiming_normal_` for convolution weights, `trunc_normal_` for linear layers and `constant_` for biases. The performance boosts up and gets close to the official version 
* **4 Aug 2022**: We accelerated the data processing. It seems that Numpy is much faster than Pytorch. Now we can train the model within 2 hours.
