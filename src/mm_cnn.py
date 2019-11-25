from main import main
if __name__ =='__main__':
    gpus = ['0','1']
    cfg = '../configs/mm_cnn.yaml'
    main(gpus, cfg)
