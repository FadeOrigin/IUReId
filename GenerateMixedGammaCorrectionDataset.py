import os;
import numpy;
import cv2;
import sys;
import math;
import time;
import multiprocessing;
import PIL.Image;
import argparse
import random;
import shutil;


def transferInputImage(inputImageIndex,filteredInputImageRGBFileNames,datasetName,inputDirectory,subInputDirectory,outputDirectory,subOutputDirectory,gammaValues,gammaIndexDictionary,displayGammaMultiplier):
    inputImageRGBFileName=filteredInputImageRGBFileNames[inputImageIndex];
    inputImageRGBFileNameInformation = inputImageRGBFileName.split("_");

    print("processing image " + inputImageRGBFileName);

    gammaLevels=len(gammaValues);

    cameraValue=int(inputImageRGBFileNameInformation[1][1:2]);

    gammaValue=gammaValues[numpy.random.randint(0,gammaLevels)];
    displayGammaValue=int(gammaValue*displayGammaMultiplier);
    finalOutputDirectory = outputDirectory  + "/" + subOutputDirectory;

    inputImageRGBFileNameInformation.insert(1, "c" + str(gammaIndexDictionary[gammaValue]));

    outputImageName = "_".join(inputImageRGBFileNameInformation);

    if gammaValue==1:
        shutil.copy(inputDirectory+"/" +subInputDirectory+ "/" + inputImageRGBFileName,finalOutputDirectory + "/" + outputImageName);
    else:
        lookUpTable = numpy.empty((1, 256), numpy.uint8);
        for i in range(256):
            lookUpTable[0, i] = numpy.clip(pow(i / 255.0, gammaValue) * 255.0, 0, 255)
        originalImage = cv2.imread(inputDirectory+"/" +subInputDirectory+ "/" + inputImageRGBFileName);
        imageAfterGammaCorrect = cv2.LUT(originalImage, lookUpTable);
        cv2.imwrite(finalOutputDirectory+"/"+ outputImageName, imageAfterGammaCorrect);

def transformAFolder(datasetName,inputDirectory,subInputDirectory,outputDirectory,subOutputDirectory,cameraFilterString,gammaValues,gammaIndexDictionary,displayGammaMultiplier):

    cameraIdList=cameraFilterString.split(",");
    inputImageRGBFileNames = os.listdir(inputDirectory+"/"+subInputDirectory);
    filteredInputImageRGBFileNames = [];
    for cameraId in cameraIdList:
        filteredInputImageRGBFileNames.extend([x for x in inputImageRGBFileNames if x.find(cameraId) > -1]);
    filteredInputImageCount=len(filteredInputImageRGBFileNames);

    useLogicalCoreCount = 36;
    batchCount = math.ceil(filteredInputImageCount / useLogicalCoreCount);
    for batchIndex in range(batchCount):
        batchStartIncluded = batchIndex * useLogicalCoreCount;
        batchEndExcluded = min(filteredInputImageCount, (batchIndex + 1) * useLogicalCoreCount);
        multiProcessingPool = multiprocessing.Pool(batchEndExcluded - batchStartIncluded);
        multiProcessingTasks = [multiProcessingPool.apply_async
                                (transferInputImage, args=(
                                    inputImageIndex
                                    , filteredInputImageRGBFileNames
                                    , datasetName
                                    , inputDirectory
                                    , subInputDirectory
                                    , outputDirectory
                                    , subOutputDirectory
                                    , gammaValues
                                    , gammaIndexDictionary
                                    , displayGammaMultiplier)
                                 ) for inputImageIndex in range(batchStartIncluded, batchEndExcluded)];
        multiProcessingResults = [task.get() for task in multiProcessingTasks];
        multiProcessingPool.close();


if __name__ == '__main__':

    parser = argparse.ArgumentParser();
    parser.add_argument('--gammaValue',default=1.0, type=float);
    opt = parser.parse_args()

    rootDirectory = os.path.dirname(os.getcwd());
    inputDirectory = "C:/Datasets/Market";
    outputDirectory = "C:/PaperData/IU-ReID/Market_LowLight_2I";
    datasetName="Market"
    cameraFilterString="c1,c2,c3,c4,c5,c6,c7,c8";

    startTime = time.time();

    gammaValues = [0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.5,2.9,3.3];
    gammaIndexDictionary={0.3:1,0.4:2,0.5:3,0.6:4,0.8:5,1.0:6,1.2:7,1.5:8,1.8:9,2.1:10,2.5:11,2.9:12,3.3:13};


    displayGammaMultiplier=100;

    #subInputDirectory="bounding_box_test";
    #subOutputDirectory="bounding_box_test";
    subInputDirectory = "query";
    subOutputDirectory = "query";
    finalOutputDirectory = outputDirectory + "/" + subOutputDirectory;
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory);
    if not os.path.exists(finalOutputDirectory):
        os.mkdir(finalOutputDirectory);
    transformAFolder(datasetName,inputDirectory,subInputDirectory, outputDirectory,subOutputDirectory, cameraFilterString, gammaValues,gammaIndexDictionary,displayGammaMultiplier);

    print("total time=" + str(round(time.time() - startTime, 3)));

