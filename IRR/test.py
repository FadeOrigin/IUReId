import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import sys;
import numpy;


def processADirectory(opt,inputSubDirectory,outputSubDirectory):
    outputDirectory=os.path.join(opt.save_root, outputSubDirectory);
    data_loader = CreateDataLoader(opt,opt.dataroot,inputSubDirectory)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    #labelFilter=["real_A","real_B","rec_A","rec_B","fake_A","fake_B"];
    labelFilter=[];
    if inputSubDirectory=="bounding_box_train":
        labelFilter = ["rec_A"];
        #labelFilter = ["fake_B"];
    else:
        labelFilter=["fake_A"]
    #labelFilter = ["fake_B"];
    # test
    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
            # break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(visuals, img_path, opt.illA, opt.illB, labelFilter, outputDirectory);

        '''
        if inputSubDirectory=="bounding_box_train":
            randomInt=numpy.random.randint(0,10);
            if randomInt==9:
                save_images(visuals, img_path, opt.illA, opt.illB, labelFilter, outputDirectory);
                print('processing (%04d)-th image... %s' % (i, img_path))
        else:
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(visuals, img_path, opt.illA, opt.illB, labelFilter,outputDirectory);
        '''

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.loadSize = 256
    opt.fineSize = 256

    #processADirectory(opt, "bounding_box_train","bounding_box_train");
    #processADirectory(opt, "bounding_box_train", "reconstruct");
    #processADirectory(opt,"query","query");
    processADirectory(opt, "bounding_box_test","bounding_box_test");


