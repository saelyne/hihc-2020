# hihc-2020
[Team 와플] Sign language recognition using [ms-tcn](https://github.com/yabufarha/ms-tcn)

## Usage
### Training
0. Include `hackathon_data_3000` folder in the root
1. run `prepare_data.py`.
- Adjust `TOP_N` to your target number of classes. Currently set to 10.
- Currently only using real data from `REAL_DATA_PATH`. Change `TARGET_DATA_PATH` to `SYN_DATA_PATH` if you want to train with synthesized vidoes.

2. run `slice_data_np.py`
3. run `train_np.py`
- USAGE: `python train_np.py --model ../output/hand.pth`
- The model and graph representing accuracy and loss will be saved in `output` folder

### Testing
1. run `test.py` with `TEST_PATH` variable set to the path to your target video.
- USAGE: `python test.py --model ../output/hand.pth`
