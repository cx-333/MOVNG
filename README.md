# MOVNG

This repo is the implementation for paper <font color=Yellow>**'MOVNG: Applied a Novel Sparse Fusion Representation into GTCN
for Pan-cancer Classification and Biomarker Identification'** </font>.

# Architecture of MOVNG
![](figures/overall_framework0.png)

# Prerequisites

* python >= 3.x
* pytorch >= 1.1.x
* PyG 
* We provide the dependency file of our experimental environment, you can install 
all dependencies by creating a new anaconda virtual environment and running :
`pip install -r requirements.txt`

# Training & Testing

## Training 

* If you want to re-train the model, you can run following commands, sequentially.
```commandline
python firststage.py --device cpu or cuda --MD train
python secondstage.py --device cpu or cuda --MD train
python finetune.py --device cpu or cuda --MD train
```

## Testing

* The final trained model can be loaded in file finetune.py, and you can test model using the command.

```commandline
python finetune.py --device cpu or cuda --MD test
```
> * Note: This paper prepared a model already trained by ourselves, which readers can see the performance of the model by the command mentioned above.
> * Other options: When you are training the model, there are some additional parameters can be changed by users. More details see the parse_args.py. 
> * Essential changes: In addition to modify the optional parameters, you should also to turn the corresponding arguments existed in firststage.py, secondstage.py and finetune.py files.


# Contact

For any questions, feel free to contact: `cxregion.@163.com`
