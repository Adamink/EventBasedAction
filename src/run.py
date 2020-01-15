from main import main
if __name__ =='__main__':
    gpus = ['1','2','3','4']
    cfg = '../configs/shuffle_feat.yaml'
    main(gpus, cfg)
