# Multistain-CycleGAN: Multi-domain stain normalization with a single model

![image principle](/docs/assets/principle.png)

Pytorch implementation of MultiStain-CycleGAN described in "Multi-domain stain normalization for digital pathology: A cycle-consistent adversarial
network approach for whole slide images". MultiStain-CycleGAN is capable of many-to-one stain translation.

This repo is based on the works of [Jun-Yan Zhu and Taesung Park et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Installation

* Clone this repo with:  

`git clone https://github.com/marjohe/multistain_cyclegan.git`

* Install the requirements:
    + Pip users: `pip install -r requirements.txt`
    + Conda users: `conda env create -f environment.yml`
  
### Minimal working example of the normalization process:
We ship a working generator trained on the CAMELYON17 dataset as described in the paper. We trained the generator to translate images to the target domain _center_0_ 
* Download the model weights from [here](https://drive.google.com/file/d/1IAPrELC73Iae13CvTxDRRuxZ-vRWzZZp/view?usp=share_link)
* Copy them into `resources/models`  
* Run `normalize.py` to normalize and plot the example images of every of the 5 centers of the CAMELYON17 dataset.

### Train your own model
* Create a root directory containing 2 directories named `trainA` and `trainB` with images of different medical centers / slides
* Pass a list of arguments to `train.py` and run it(`--dataroot` and `--name` are required)
    + `dataroot`: path/to/train_dir containing `trainA` and `trainB`
    + `name`: experiment name
    
* For visualization of the train process a visdom server is required:
    + To install visdom: `pip install visdom`
    + To start the server: `python -m visdom.server`
    + Click [http://localhost:8097](http://localhost:8097) to view the results  
    + Visdom can be disabled by passing `--display_id 0`
* For troubleshooting look here: [Frequently Asked Questions](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md)

### Tiling your own slides
We recommend using our publicly available pipeline for tiling whole slide images that can be found here: [WSI Preprocessing](https://github.com/DBO-DKFZ/wsi_preprocessing)

### Citation
If you use this work for your research please cite:

 @misc{hetz2023multidomain,
      title={Multi-domain stain normalization for digital pathology: A cycle-consistent adversarial network for whole slide images}, 
      author={Martin J. Hetz and Tabea-Clara Bucher and Titus J. Brinker},
      year={2023},
      eprint={2301.09431},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
