# Person Search Research

This codebase is a framework of person search research and provides serveral strong person search models with simple design.

This codebase contains some methods in one of our unpublished paper and [ACAE paper](https://arxiv.org/abs/2111.14316).

## train

1. **setup dataset**: put CUHK-SYSU and PRW under folder `data/` and ensure the paths are `data/cuhk-sysu` and `data/prw` seperately.

```bash
mkdir data
sudo ln -s <cuhk-sysu-target-folder> data/cuhk-sysu
sudo ln -s <prw-target-folder> data/prw
```

2. **train:** we provide different trainer for different circumstances, but each trainer accepts only one argument `--cfg` to specify the path of target config file.

```bash
# for common tranining without graph modules
python -m models.trainer --cfg <cfg-path>
# for training RetinaNet or FCOS based model without graph modules
python -m models.iter_trainer --cfg <cfg-path>
# for training with ACAE
python -m models.graph_trainer --cfg <cfg-path> 
# for training with ACAE but freeze the person search model
python -m models.freeze_trainer --cfg <cfg-path>
```

## test

This codebase provides three ways to evaluate the performance of person search models.

- **Common way.**
- **CMM way:** an re-ranking like evaluation method, which could be adapted on all person search models if providing the features of query and gallery.
- **ACAE way:** another re-ranking like evaluation method, however only for models with trained ACAE module.

CMM way and ACAE way both take co-walkers information into consideration and in most circumstances, they could improve the baseline person search performance without much modification.

To evaluate, run the following commands, where `<exps_dir>` is the saved folder, and `<eval-config-path>` is the path of an evaluation config file.

If `<eval-config-path>` is not specified, the priority of evaluation file is: `eval.yml` > `config.yml` under `<exps_dir>`.

```bash
# common way and cmm way
python -m evaluation.eval_defaults <exps_dir> --eval-config <eval-config-path>
# acae way
python -m evaluation.eval_graph <exps_dir> --eval-config <eval-config-path>
```

__NOTE: check scripts under `bin/` for more details and examples.__

## demo

This codebase also includes a simple person search system under `demo/` with a concise GUI dashboard implemented by Tkinter.

![GUI](doc/demo_gui.png)

To run the demo:

1. download the pretrained models and serialized pickles which store the extracted detected boxes and persons' features of CUHK-SYSU.

2. go to the `choose_model_dataset` function in `demo/search_tools.py` to specify the base root of checkpoint in `exp_dir` and the path of serialized pickle in `pkl_path`.

3. run the demo with

```bash
python -m demo.main
```

4. usage of the gui program: Select the image file first; then Select Person by pointing out the bounding box of the target query person (Once decided, close the prompt window); Select search options and go search!

More about the demo:

- Currently, this gui program runs well in Ubuntu18.04, but is **not** tested on Windows. There might be differences between the two OS about events binding in Tkinter.
- Feel free to change the options of searching in the gui program.

## pretrained model

- All pretrained models are based on ResNet-50 backbone.
- Each `checkpoint` zipfile contains three files: checkpoint, `config.yml` and `eval.yml` (only for evaluation).
- Results are all trained on one Nvidia RTX2080Ti.

| name | detection-baseline | dataset | det-ap | det-recall | reid-map | reid-top-1 | checkpoint | pkl |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| baseline | Faster-RCNN | CUHK-SYSU |  | | | | | |
| baseline | Faster-RCNN | PRW |  | | | | | |
| baseline | FPN | CUHK-SYSU |  | | | | | |
| baseline | RetinaNet | CUHK-SYSU |  | | | | | |
| baseline | FCOS | CUHK-SYSU |  | | | | | |
| ACAE | Faster-RCNN | CUHK-SYSU |  | | | | | |
| ACAE | Faster-RCNN | PRW |  | | | | | |
| ACAE | FPN | CUHK-SYSU |  | | | | | |
| ACAE | RetinaNet | CUHK-SYSU |  | | | | | |
| ACAE | FCOS | CUHK-SYSU |  | | | | | |

Moreover:

- Baseline models on FPN/FCOS/RetinaNet are experimental, so the overall search performance are not finetuned to the best.

## TODO/Update

- [ ] Bugs in moving/resize view of pictures in the GUI program.
- [x] Optimize the dashboard of the GUI program, especially removing unrelated inplace images and adding more description.

## Reference

This codebase is inspired by the following person search codebases:

- NAE
- SeqNet
