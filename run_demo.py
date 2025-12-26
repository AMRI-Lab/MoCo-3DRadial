import train
import utils

if __name__ == '__main__':

    config_path = 'config.yaml'
    config = utils.load_config(config_path)

    train.train(config)


