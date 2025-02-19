# SALS (ICLR 2025)
## Preparation
Install **ngpmesh** to do the intersection detection.   
```
cd third_party/ngpmesh
pip install .
cd ../..
```
Generate E-DC data for surface extraction. It will take a while, please be patient. (Welcome to discuss more efficient solutions with us!!!)
```
python neural_shape_representation/diff_emc.py
```
## Neural Shape Representation
Utilize an MLP to model a shape, and output the extracted surface.
```
cd neural_shape_representation
python overfit_main.py --filename <input_mesh> --ckpt_path <save_path>
```

## Point Cloud Surface Reconstruction
Prepare dataset.
```
python preprocess_data.py
```
Train the network.
```
python train.py --data_path <dataset path>
```
Evaluate the network
```
python eval.py --config_path <path to config file>
```
