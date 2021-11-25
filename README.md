<center>
<img src=".github/mesa_banner.jpg"style="zoom:100%;" />
</center>


# A Memory-saving Training Framework for Transformers

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official PyTorch implementation for [Mesa: A Memory-saving Training Framework for Transformers](https://arxiv.org/abs/2111.11124).

By [Zizheng Pan](https://scholar.google.com.au/citations?user=w_VMopoAAAAJ&hl=en), [Peng Chen](https://scholar.google.com/citations?user=Hoh9p_kAAAAJ&hl=en), [Haoyu He](https://scholar.google.com/citations?user=aU1zMhUAAAAJ&hl=en), [Jing Liu](https://sites.google.com/view/jing-liu/首页), [Jianfei Cai](https://scholar.google.com/citations?user=N6czCoUAAAAJ&hl=en) and  [Bohan Zhuang](https://sites.google.com/view/bohanzhuang).


<center>
<img src=".github/framework_v1.jpg" style="zoom:100%;" />
</center>

## Installation

1.  Create a virtual environment with anaconda.

       ```bash
       conda create -n mesa python=3.7 -y
       conda activate mesa
       
       # Install PyTorch, we use PyTorch 1.7.1 with CUDA 10.1 
       pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
       
       # Install ninja
       pip install ninja
       ```

2.  Build and install Mesa.

       ```bash
       # clone this repo
       git clone https://github.com/zhuang-group/Mesa
       # build
       cd Mesa/
       # You need to have an NVIDIA GPU
       python setup.py develop
       ```



## Usage
 1. Prepare your policy and save as a text file, e.g. `policy.txt`.
    ```bash
    on gelu: # layer tag, choices: fc, conv, gelu, bn, relu, softmax, matmul, layernorm
        by_index: all # layer index
        enable: True # enable for compressing
        level: 256 # we adopt 8-bit quantization by default
        ema_decay: 0.9 # the decay rate for running estimates
        
        by_index: 1 2 # e.g. exluding GELU layers that indexed by 1 and 2.
        enable: False
    ```
 2. Next, you can wrap your model with Mesa by:

    ```python
    import mesa as ms
    ms.policy.convert_by_num_groups(model, 3)
    # or convert by group size with ms.policy.convert_by_group_size(model, 64)
    
    # setup compression policy
    ms.policy.deploy_on_init(model, '[path to policy.txt]', verbose=print, override_verbose=False)
    ```

    **That's all you need to use Mesa for memory saving.** 
    
    Note that `convert_by_num_groups` and `convert_by_group_size` only recognize `nn.XXX`, if your code has functional operations, such as `Q@K` and `F.Softmax`, you may need to manually setup these layers.  For example:

    ```python
    import mesa as ms
    # matrix multipcation (before)
    out = Q@K.transpose(-2, -1)
    # with Mesa
    self.mm = ms.MatMul(quant_groups=3)
    out = self.mm(q, k.transpose(-2, -1))

    # sofmtax (before)
    attn = attn.softmax(dim=-1)
    # with Mesa
    self.softmax = ms.Softmax(dim=-1, quant_groups=3)
    attn = self.softmax(attn)
    ```

 3. You can also target one layer by:

    ```python
    import mesa as ms
    # previous 
    self.act = nn.GELU()
    # with Mesa
    self.act = ms.GELU(quant_groups=[num of quantization groups])
    ```
    

## Demo projects for DeiT and Swin

We provide demo projects to replicate our results of training DeiT and Swin with Mesa, please refer to [DeiT-Mesa](https://github.com/HubHop/deit-mesa) and [Swin-Mesa](https://github.com/HubHop/swin-mesa).


## Results on ImageNet

| Model               | Param (M) | FLOPs (G) | Train Memory (MB) | Top-1 (%) |
| ------------------- | --------- | --------- | ------------ | --------- |
| DeiT-Ti             | 5         | 1.3       | 4,171         | 71.9      |
| **DeiT-Ti w/ Mesa** | 5         | 1.3       | **1,858**     | **72.1**  |
| DeiT-S              | 22        | 4.6       | 8,459         | 79.8      |
| **DeiT-S w/ Mesa**  | 22        | 4.6       | **3,840**     | **80.0**    |
| DeiT-B              | 86        | 17.5      | 17,691        | 81.8      |
| **DeiT-B w/ Mesa**  | 86        | 17.5      | **8,616**     | **81.8**  |
| Swin-Ti             | 29        | 4.5       | 11,812        | 81.3      |
| **Swin-Ti w/ Mesa** | 29        | 4.5       | **5,371**     | **81.3**  |
| PVT-Ti              | 13        | 1.9       | 7,800         | 75.1      |
| **PVT-Ti w/ Mesa**  | 13        | 1.9       | **3,782**     | 74.9      |

> Memory footprint at training time is measured with a batch size of 128 and an image resolution of 224x224 on a single GPU.


## Citation
If you find our work interesting or helpful to your research, please consider citing Mesa.

```
@article{pan2021mesa,
      title={Mesa: A Memory-saving Training Framework for Transformers}, 
      author={Zizheng Pan and Peng Chen and Haoyu He and Jing Liu and Jianfei Cai and Bohan Zhuang},
      journal={arXiv preprint arXiv:2111.11124}
      year={2021}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/zhuang-group/Mesa/blob/main/LICENSE) file.


## Acknowledgments

This repository has adopted part of the quantization codes from [ActNN](https://github.com/ucbrise/actnn), we thank the authors for their open-sourced code.

