import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader # our data_loader
import numpy as np
from trijoint import im2recipe, norm
import pickle
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)

np.random.seed(opts.seed)

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda',0))

def main():
   
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)


    print("=> loading checkpoint '{}'".format(opts.model_path))
    if device.type=='cpu':
        checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(opts.model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    
    # preparing test loader 
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
 	    transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            True,
        ]),data_path=opts.data_path,sem_reg=opts.semantic_reg,partition='test'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Test loader prepared.')

    # run test
    test(test_loader, model)

def test(test_loader, model):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input_var = list() 
        for j in range(len(input)):
            if j == 0:
                continue
            input_var.append(input[j].to(device))
        target_var = list()
        for j in range(len(target)-2): # we do not consider the last two objects of the list
            if j == 0:
                continue
            target_var.append(target[j].to(device))

        # recipe embedding
        recipe_emb = model.table([model.stRNN_(input_var[1], input_var[2]), model.ingRNN_(input_var[3], input_var[4]) ],1) # joining on the last dim
        recipe_emb = model.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i==0:
            data1 = recipe_emb.data.cpu().numpy()
            data2 = target[-2]
            data3 = target[-1]
        else:

            data1 = np.concatenate((data1,recipe_emb.data.cpu().numpy()),axis=0)
            data2 = np.concatenate((data2,target[-2]),axis=0)
            data3 = np.concatenate((data3,target[-1]),axis=0)


    with open(opts.path_results+'rec_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(opts.path_results+'img_ids.pkl', 'wb') as f:
        pickle.dump(data2, f)
    with open(opts.path_results+'rec_ids.pkl', 'wb') as f:
        pickle.dump(data3, f)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
