from main import main
if __name__ =='__main__':
    gpus = ['0','1','6','8']
    cfg = '../configs/pose_7500_new.yaml'
    main(gpus, cfg)
