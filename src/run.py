from main import main
if __name__ =='__main__':
    gpus = ['2','3']
    cfg = '../configs/final/frame_feat_heat.yaml'
    main(gpus, cfg)
