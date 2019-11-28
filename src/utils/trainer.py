import torch
from tqdm import tqdm
from .timer import get_fmt_time, get_time, fmt_elapsed_time

def train_one_epoch(classifier, dataloader, optimizer, criterion, metric_func):
    train_all, train_loss, train_metric = (0,)* 3
    data_time, model_time = (0,) * 2
    classifier.train()
    with tqdm(total = len(dataloader)) as pbar:
        t1 = get_time()
        for i, data in enumerate(dataloader):
            pbar.update(1)
            t2 = get_time()
            data_time += t2 - t1

            source, target = data
            source, target = source.cuda(), target.cuda()
            pred = classifier(source)

            loss = criterion(pred, target)
            metric = metric_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_all += source.size()[0]
            train_loss += loss.item()
            train_metric += metric.item()
            
            t1 = get_time()
            model_time += t1 - t2

    return train_loss / train_all, train_metric / train_all, data_time, model_time

def test_one_epoch(classifier, dataloader, criterion, metric_func):
    test_loss, test_all, test_metric = (0,) * 3
    data_time, model_time = (0,) * 2
    classifier.eval()
    with torch.no_grad(), tqdm(total = len(dataloader)) as pbar:
        t1 = get_time()
        for i, data in enumerate(dataloader):
            pbar.update(1)
            t2 = get_time()
            data_time += t2 - t1

            source, target = data
            source, target = source.cuda(), target.cuda()
            pred = classifier(source)

            loss = criterion(pred, target)
            metric = metric_func(pred, target)

            test_all += source.size()[0]
            test_loss += loss.item()
            test_metric += metric.item()

            t1 = get_time()
            model_time += t1 - t2

    return test_loss / test_all, test_metric / test_all, data_time, model_time

def train(args, model, train_dataloader, test_dataloader, optimizer, scheduler,
 criterion, metric, logger, board, test_interval = 1, max_metric = False):

    def better(a, b):
        if max_metric:
            return a > b
        else:
            return a < b

    best_metric, best_epoch = (float('-inf'), 0) if max_metric else (float('inf'), 0)
    test_loss, test_metric, data_time, model_time = \
     test_one_epoch(model, test_dataloader, criterion, metric)
    logger("Initially, test_loss: {:.4f}, test_metric: {:.4f}, data_time: {:s}, model_time: {:s}".
     format(test_loss, test_metric, fmt_elapsed_time(data_time), fmt_elapsed_time(model_time)))
    
    scheduler_after_epoch = args.stages[-1]['epoch'] if hasattr(args, 'stages') else 0
    for epoch in range(args.epochs):
        if hasattr(args, 'stages'):
            for _ in args.stages:
                e = _['epoch']
                operation  = _['operation']
                if epoch==e:
                    logger(operation)
                    eval('model.module.' + operation)

        train_loss, train_metric, data_time, model_time = \
         train_one_epoch(model, train_dataloader, optimizer, criterion, metric)
        output = 'Train: Epoch {:>3d} Time {:s} Datatime {:s} ModelTime {:s} Loss {:.4f} Metric {:.4f}'.format(
         epoch, get_fmt_time(), fmt_elapsed_time(data_time), fmt_elapsed_time(model_time), train_loss, train_metric)
        logger(output)
        board.add_scalars('loss',{'train_loss':train_loss}, epoch)
        board.add_scalars('metric',{'train_metric':train_metric}, epoch)

        if epoch >= scheduler_after_epoch:
            if scheduler.__class__.__name__=='ReduceLROnPlateau':
                scheduler.step(train_loss)
            else:
                scheduler.step()

        if epoch % test_interval == 0:
            test_loss, test_metric, data_time, model_time = test_one_epoch(model, test_dataloader, criterion, metric)
            output = 'Test : Epoch {:>3d} Time {:s} Datatime {:s} ModelTime {:s} Loss {:.4f} Metric {:.4f}'.format(
             epoch, get_fmt_time(), fmt_elapsed_time(data_time), fmt_elapsed_time(model_time), test_loss, test_metric)
            logger(output)
            board.add_scalars('loss',{'test_loss':test_loss}, epoch)
            board.add_scalars('metric', {'test_metric':test_metric}, epoch)

            if better(test_metric, best_metric):
                print("Epoch: {:>3d} Saving Model".format(epoch))
                best_metric, best_epoch = test_metric, epoch
                torch.save(model.state_dict(), args.model_savepth)
                torch.save(optimizer.state_dict(), args.optim_savepth)

    return best_metric, best_epoch
