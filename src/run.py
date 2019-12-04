from main import main
if __name__ =='__main__':
    gpus = ['0','1','2','3']
    cfg = '../configs/action_7500.yaml'
    main(gpus, cfg)
