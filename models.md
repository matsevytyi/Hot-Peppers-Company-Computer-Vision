# Models

Models are located in mamba-vision-ours. You can load 2 types of models from this module:

1. Base object detection model, consisting of **mamba-vision backbone** (`base_model.py`) and **yolov11 detection head** and **neck** (`yolov11detection/head.py` and `yolov11detection/neck.py`)
2. MoE object detection model, acts like a wrapper around base model. It introduces domain-specific LorA adapters (`adaopters/lora.py`) and router (`adapters/router.py`) that plugs them in. Router analyzes whole image and assigns one LoRA adapter to it rather than different adapters to different or same patches of same image. It is optimized for the inference by loading all adapters in memory and swapping them in runtime.

Model setup on training and inference is done through `configs/training/*.yaml`. 

There are pre-trained model checkpoints at `checkpoints/*`, grouped into router, base and lora. You can swap the freely by modifying config file. Also, you can train your own models or evaluate throgh `notebooks/*.ipynb`.

LoRA adapters work differently – check for name. It must inlude dataset name (e.g. `bdd_100k_day`) and scenario (e.g. `blanket`) under which it was trained. As of our latest update, we consider three scenarios:

1. **'Blanket' scenario** – plugs in LoRA adapters to each FC layer (only in backbone)
2. **'Top-1' scenario** – plugs in LoRA adapter to backbone layer with highest variance (usually around 3rd level)
3. **'Top-k' scenario** – plugs in LoRA adapter to a set of backbone layers thaty have substansially higher variance compared to others (e.g. std of 40 compared to 5 in other layers on the domain shifts)

To determine exact Top-1 or Top-k layers, feel free to run `notebooks/01_variance_analysis.ipynb`.

## Important Notes:

**Mamba-vision backbone** is changed from original implementation by NVidia, as the latter implementation does not account for (1) feature extraction and (2) changing deltas.

We recommend using our fork that does so and was used by us duringh training and testing – https://github.com/matsevytyi/MambaVisionReengineering 