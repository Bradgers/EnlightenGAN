import time
from tqdm import tqdm
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
output_dir = os.path.join("./ablation/", opt.name)
metric_name = os.path.join(output_dir, "metric.txt")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
total_epoch = int(opt.which_epoch)
# test
print(len(dataset))
for epoch in tqdm(range(total_epoch)):
    t1 = time.time()
    opt.which_epoch = epoch + 1
    model = create_model(opt)
    psnr, niqe = 0, 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
        metrics = model.evaluate()
        psnr += metrics["psnr"]
        niqe += metrics["niqe"]
        # img_path = model.get_image_paths()
        # print('process image... %s' % img_path)
    print("epoch %d: psnr: %f, niqe: %f\n" %(opt.which_epoch, psnr / len(dataset), niqe / len(dataset)))
    with open(metric_name, 'a') as metric_file:
        metric_file.write('epoch %d: psnr: %f niqe: %f time: %f\n' %(opt.which_epoch, psnr / len(dataset), niqe / len(dataset), time.time() - t1))