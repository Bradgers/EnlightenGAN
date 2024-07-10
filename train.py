import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

opt = TrainOptions().parse()
config = get_config(opt.config)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
metric_name = os.path.join(opt.checkpoints_dir, opt.name, 'metric.txt')

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    # ssim, psnr = 0, 0
    psnr, niqe = 0, 0
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters(epoch)
        # metrics = model.evaluate()
        # psnr += metrics["psnr"]
        # niqe += metrics["niqe"]
        # print('epoch %d: ssim: %f psnr: %f' %(epoch, ssim, psnr))
        # print(ssim, psnr)
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    
    # # mean_ssim = ssim / dataset_size
    # mean_psnr = psnr / dataset_size
    # mean_niqe = niqe / dataset_size
    # print('epoch %d: psnr: %f niqe: %f' %(epoch, mean_psnr, mean_niqe))
    # with open(metric_name, "a") as metric_file:
    #     metric_file.write('epoch %d: psnr: %f niqe: %f \n' %(epoch, mean_psnr, mean_niqe))

    if opt.new_lr:
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()
