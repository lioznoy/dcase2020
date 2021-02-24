# dcase2020
Acoustic scene classification - DCASE 2020 challenge task1

## Data collection
1) Clone DCASE challenge baseline code from https://github.com/toni-heittola/dcase2020_task1_baseline

2) Install all repository requirements.

3) Download challenge task 1a dataset: `python task1a.py --download_dataset DATASET_PATH`

4) Download challenge task 1b dataset: `python task1b.py --download_dataset DATASET_PATH`

## Clone repository
1) `git clone https://github.com/lioznoy/dcase2020.git` or download all scripts.

2) Run `pip install -r requirements.txt`

## Create features
Run `python create_features.py -n <[3 \ 10]> --data_path DATA_PATH`

Add `--force` to inforce create features from scratch, otherwise will skip existing features.

Add `--fix` to fix defected features pkl files.

The script will create a features directory named mel_features_3d containing all the features matrices pkl files and augmentations.

## Train model

Run `python main.py` with the following parameters:

| Argument            |                     | Description                                                  | Defualt value                                                                  |
| --------------------| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `-h`                | `--help`            | ##########################################|                                                                                |
| `-e`                | `--epochs`          | Number of epochs for training model                          | 100                                                                            |
| `-b`                | `--batch_size`      | Batch size                                                   | 32                                                                             |                                                                 
| `-l`                | `--lr`              | Learning rate                                                | 0.001                                                                          |
| `-w`                | `--weights`         | Load existing weights for the model                          | None                                                                           |
| `--data_dir_10`     | `--data_dir_10`     | Path for task 1a (10 labels) data directory                  | ../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/audio            |
| `--data_dir_3`      | `--data_dir_3`      | Path for task 1b (3 labels) data directory                   | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/audio            |
| `--features_dir_10` | `--features_dir_10` | Path for task 1a (10 labels) features directory              | ../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/mel_features_3d  |
| `--features_dir_3`  | `--features_dir_3`  | Path for task 1b (3 labels) features directory               | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/mel_features_3d  |
| `--folds_dir_10`    | `--folds_dir_10`    | Path for task 1a (10 labels) folds split directory           | ../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup |
| `--folds_dir_3`     | `--folds_dir_3`     | Path for task 1a (3 labels) folds split directory            | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup |
| `--output_dir`      | `--output_dir`      | Directory to save loss & score figures and logs txt files    | outputs                                                                        |
| `--dir_checkpoint`  | `--dir_checkpoint`  | Directory to save weights after each epoch                   | checkpoints                                                                    |
| `--n_classes`       | `--n_classes`       | Number of classes 3 or 10                                    | 10                                                                             |
| `--gpu`             | `--gpu`             | GPU number to run (in case of multiple gpus)                 | 0                                                                              |
| `--backbone_10`     | `--backbone_10`     | Backbone for CNN classifier 10 classes                       | resnet18                                                                       |
| `--backbone_3`      | `--backbone_3`      | Backbone for CNN classifier 3 classes                        | mobilenetV2                                                                    |
| `--setup`           | `--setup`           | Setup to run single_path / fcnn / two_path / two_path_fcnn   | two_path                                                                       |
| `--augmentations`   | `--augmentations`   | without / no_impulse / all                                   | all                                                                            |

Example:
`-b 32 -e 100 -l 0.001 --setup mobilenet_v2 --output_dir output_3class_mobilenet_v2 --dir_checkpoint checkpoints_3class_mobilenet_v2 --n_classes 3`

## Evaluate model

Run `python evaluate_model.py` with the following parameters:

| Argument            |                     | Description                                                  | Defualt value                                                                  |
| --------------------| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `-h`                | `--help`            | ##########################################                   |                                                                                |
| `-b`                | `--batch_size`      | Batch size                                                   | 32                                                                             |                                                                 
| `--weights`         | `--weights`         | weights .pth file to evaluate                                | None                                                                           |
| `--data_dir`        | `--data_dir`        | Path for data directory                                      | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/audio            |
| `--features_dir`    | `--features_dir`    | Path for features directory                                  | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/mel_features_3d  |                          
| `--test_csv`        | `--test_csv`        | Path for test csv file                                       | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/fold1_evaluate.csv |
| `--folds_dir_3`     | `--folds_dir_3`     | Path for task 1a (3 labels) folds split directory            | ../datasets/TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup |
| `--output_dir`      | `--output_dir`      | Directory to save confusion matrices                         | output_test                                                                    |
| `--n_classes`       | `--n_classes`       | Number of classes 3 or 10                                    | 10                                                                             |
| `--gpu`             | `--gpu`             | GPU number to run (in case of multiple gpus)                 | 0                                                                              |
| `--backbone_10`     | `--backbone_10`     | Backbone for CNN classifier 10 classes                       | resnet18                                                                       |
| `--backbone_3`      | `--backbone_3`      | Backbone for CNN classifier 3 classes                        | mobilenetV2                                                                    |
| `--setup`           | `--setup`           | Setup to run single_path / fcnn / two_path / two_path_fcnn   | two_path                                                                       |

Example:
`python evaluate_model.py --weights /mnt/disks/nlioz/DCASE2021/dcase2020/checkpoints_3class_mobilenet_v2_2/CP_epoch77.pth -b 32 --setup mobile_v2 --n_classes 3`