"""
    Train a IR-GAN model:
    python train.py --dataroot ./datasets/KAIST --name KAIST_IRGAN --model IR-GAN \
        --direction BtoA --tevnet_weights path/to/tevnet/checkpoint.pth
"""

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from TeVNet.utils import TeVloss

if __name__ == '__main__':
    opt = TrainOptions().parse()         # get training options
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()     # create a dataset given opt.dataset_mode and other options
    dataset_size = len(data_loader)       # get the number of images in the dataset.
    print('#training images = %d' % dataset_size)

    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)    # create a visualizer that display/save images and plots
    total_steps = 0                 # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        print("epoch", epoch)
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()    # timer for entire epoch
        iter_data_time = time.time()      # timer for data loading per iteration
        epoch_iter = 0                    # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
