To train run:

```
python train.py
```

which will train an save the result to `chkpt.pth`, and a loss plot to `losses.png`.

To visualize run:

```
python viz.py
```

which will load `chkpt.pth` and visualize the learned inverse kinematics model. There is also a `sanity_check` function in `viz.py` which may be useful.