### Preparing datasets.
```
data
|—————— roidbs
|      |—————— train2017_det_gt_roidb.pkl
|      |—————— val2017_det_gt_roidb.pkl
|—————— images_lst_rec
|      |—————— train2017.lst
|      |—————— train2017.rec
|      |—————— train2017.idx
|—————— annotations
|      |—————— instances_train2017.json
|      |—————— instances_val2017.json
```

### Train on custom datasets.
Here is an example for roidb information.
```
[
    {
        'image': 'a.jpg',
        'width': 1000,
        'height': 600,
        'boxes': <np.ndarray> (n, 4),
        'gt_classes': <np.ndarray> (n, ),
        'gt_masks': <np.ndarray> (n, m),
        'gt_keypoints': <np.ndarray> (n, p)
    },
    ...
]
```
- First, to generate roidb for the datasets.
- Second, to generate *.lst for split train,val,test datasets.
- Third, to generate *.rec and *.idx.

#### Generating list file.
- demo_generate_lst.py
```
integer_image_index \t label_index \t path_to_image
```
```
data
|—————— images_lst_rec
|      |—————— train2017.lst
|      |—————— val2017.lst
|      |—————— test2017.lst
```
#### Generating rec file.
- demo_make_rec.py

```
data
|—————— images_lst_rec
|      |—————— train2017.rec
|      |—————— val2017.rec
|      |—————— test2017.rec
|      |—————— train2017.idx
|      |—————— val2017.idx
|      |—————— test2017.idx
```
#### Show rec results for visualization.
- demo_read_rec.py