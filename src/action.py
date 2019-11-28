from main import main
if __name__ =='__main__':
    gpus = ['4','5','6','7']
    cfg = '../configs/action_gru_1_afterfc.yaml'
    main(gpus, cfg)
