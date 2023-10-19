# μP ResNet
This folder contains the source code for our experiment on ResNet on CIFAR10, which also serves as an example usage of `mup`.

## Save Model Base Shapes
To train a μP model, one needs to first specify the base shapes. To save base shapes info, run, for example,
```
python main.py --save_base_shapes resnet18.bsh --width_mult 1
```

## Verify Implementation with Coordinate Check
Before we scale up and start training, it is recommended to check the size of activation coordinates as model width increases. We have integrated such a test in this example using the helper functions in `mup`; you can simply run:

```bash
# for SGD
python main.py --load_base_shapes resnet18.bsh --optimizer sgd --lr 0.1 --coord_check
# for Adam
python main.py --load_base_shapes resnet18.bsh --optimizer adam --lr 0.001 --coord_check
```
You should find the generated plots under `./coord_checks`, which show stable coordinate sizes under μP, e.g., 

![](coord_checks/μp_resnet18_adam_coord.png)

and growing sizes under SP, e.g.,

![](coord_checks/sp_resnet18_adam_coord.png)


## Start Training
Having verified our implementation of μP, we can scale up our model and train using the same hyperparameters used for the small model and expect that the wider model performs better on the training data and that the optimal hyperparameters transfer.
```bash
# for SGD
python main.py --width_mult 2 --optimizer musgd
# for Adam
python main.py --width_mult 2 --optimizer muadam
```

Note that if you do not specify `--load_base_shapes`, the script will default to training a SP model.