# Illumination Unification for Person Re-Identification

Code for reproducing the results of our Illumination Unification for Person Re-Identification paper.

## Environment

The code has been tested on Pytorch 1.7.1 and Python 3.8.

## Train and test

**IRR
python train.py --dataroot ./datasets/Market_GammaAdaptationGANTrain_100to30 --name Market_GammaAdaptationGANTrain_100to30 --illA 100 --illB 30
python train.py --dataroot ./datasets/Market_GammaAdaptationGANTrain_100to40 --name Market_GammaAdaptationGANTrain_100to40 --illA 100 --illB 40
python train.py --dataroot ./datasets/Market_GammaAdaptationGANTrain_100to50 --name Market_GammaAdaptationGANTrain_100to50 --illA 100 --illB 50
...
python train.py --dataroot ./datasets/Market_GammaAdaptationGANTrain_100to330 --name Market_GammaAdaptationGANTrain_100to330 --illA 100 --illB 330

**IE
python main.py -d market --logs-dir logs/market --dataVersion Market_Mixed_G30to330 --arch resnet50 --features 48

**re-ID
python main.py -d market --logs-dir logs/Market --camstyle 32 --epochs 50 --dataVersion Market_Mixed_G30to330_TestRestore
