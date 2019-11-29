from main import main
if __name__ =='__main__':
    gpus = ['4','5','6','7']
    # cfg = '../configs/pose7500.yaml'
    cfg = '../configs/pose7500_test.yaml'
    main(gpus, cfg)
