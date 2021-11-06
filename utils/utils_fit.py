import torch
import torch.nn as nn
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor)

            optimizer.zero_grad()
            outputs = nn.Sigmoid()(model_train(images))
            output  = loss(outputs, targets)

            output.backward()
            optimizer.step()

            with torch.no_grad():
                equal       = torch.eq(torch.round(outputs), targets)
                accuracy    = torch.mean(equal.float())

            total_loss      += output.item()
            total_accuracy  += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_step_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = torch.from_numpy(targets_val).type(torch.FloatTensor).cuda()
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = torch.from_numpy(targets_val).type(torch.FloatTensor)
                optimizer.zero_grad()
                outputs = nn.Sigmoid()(model_train(images_val))
                output  = loss(outputs, targets_val)

                equal       = torch.eq(torch.round(outputs), targets_val)
                accuracy    = torch.mean(equal.float())

            val_loss            += output.item()
            val_total_accuracy  += accuracy.item()

            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
