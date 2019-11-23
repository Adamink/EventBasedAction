import torch
from tqdm import tqdm
from .timer import get_fmt_time

def train_one_epoch(classifier, dataloader, optimizer, criterion, metric_func):
    train_all, train_loss, train_metric = (0,)* 3
    classifier.train()
    # length = len(dataloader)
    # for i in tqdm(range(length)):
    #     data = dataloader[i]
    for i, data in tqdm(enumerate(dataloader)):
        source, target = data
        source, target = source.float().cuda(), target.float().cuda()
        pred = classifier(source)

        loss = criterion(target, pred)
        metric = metric_func(target, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_all += source.size()[0]
        train_loss += loss.item()
        train_metric += metric.item()

    return train_loss / train_all, train_metric / train_all

def test_one_epoch(classifier, dataloader, criterion, metric_func):
    test_loss, test_all, test_metric = (0,) * 3
    classifier.eval()
    with torch.no_grad():
        # length = len(dataloader)
        # for i in tqdm(range(length)):
        #     data = dataloader[i]
        for i, data in tqdm(enumerate(dataloader)):
            source, target = data
            source, target = source.float().cuda(), target.float().cuda()
            pred = classifier(source)

            loss = criterion(target, pred)
            metric = metric_func(target, pred)

            test_all += source.size()[0]
            test_loss += loss.item()
            test_metric += metric.item()

    return test_loss / test_all, test_metric / test_all

def train(args, model, train_dataloader, test_dataloader, optimizer, scheduler,
 criterion, metric, logger, board, test_interval = 1, max_metric = True):

    def better(a, b):
        if max_metric:
            return a > b
        else:
            return a < b

    best_metric, best_epoch = (float('inf'), 0) if max_metric else (float('-inf'), 0)
    test_loss, test_metric = test_one_epoch(model, test_dataloader, criterion, metric)
    logger("Before train, test_loss: {:.4f}, test_metric: {:.4f}".format(test_loss, test_metric))
    for epoch in range(args.epochs):
        train_loss, train_metric = train_one_epoch(model, train_dataloader, optimizer, criterion, metric)
        output = 'Train: Epoch {:>3d} Time {:s} Loss {:.4f} Metric {:.4f}'.format(
            epoch, get_fmt_time(), train_loss, train_metric)
        logger(output)
        board.add_scalars('loss',{'train_loss':train_loss}, epoch)
        board.add_scalars('metric',{'train_metric':train_metric}, epoch)

        if scheduler.__class__.__name__=='ReduceLROnPlateau':
            scheduler.step(train_loss)
        else:
            scheduler.step()

        if epoch % test_interval == 0:
            test_loss, test_metric = test_one_epoch(model, test_dataloader, criterion, metric)
            output = 'Test : Epoch {:>3d} Time {:s} Loss {:.4f} Metric {:.4f}'.format(
            epoch, get_fmt_time(), test_loss, test_metric)
            logger(output)
            board.add_scalars('loss',{'test_loss':test_loss}, epoch)
            board.add_scalars('metric', {'test_metric':test_metric}, epoch)

            if better(test_metric, best_metric):
                best_metric, best_epoch = test_metric, epoch
                torch.save(model.state_dict(), args.model_savepth)
                torch.save(model.state_dict(), args.optim_savepth)

    return best_metric, best_epoch
