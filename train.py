from nets.siamese import siamese
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.siamese_training import Generator
from keras.optimizers import Adam, SGD
import os

import tensorflow as tf

def get_image_num(path):
    num = 0
    train_path = os.path.join(path, 'images_background')
    for character in os.listdir(train_path):
        # 在大众类下遍历小种类。
        character_path = os.path.join(train_path, character)
        num += len(os.listdir(character_path))

    return num

if __name__ == "__main__":
    input_shape = [105,105,1]
    dataset_path = "./"
    log_dir = "logs/"

    model = siamese(input_shape)
    model.summary()

    model_path = r"logs\ep044-loss0.441-val_loss0.417.h5"
    model.load_weights(model_path, by_name=True,skip_mismatch=True)
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    train_ratio = 0.9
    images_num = get_image_num(dataset_path)
    train_num = int(images_num*0.9)
    val_num = int(images_num*0.1)
    
    if True:
        # 交叉熵
        Batch_size = 20
        Lr = 1e-5

        model.compile(loss = "binary_crossentropy",
                optimizer = Adam(lr=Lr),
                metrics = ["binary_accuracy"])
        print('Train with batch size {}.'.format(Batch_size))

        gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
        # 开始训练
        model.fit_generator(gen.generate(True),
                steps_per_epoch=max(1,train_num//Batch_size),
                validation_data=gen.generate(True),
                validation_steps=max(1,val_num//Batch_size),
                epochs=700,
                initial_epoch=39,
                callbacks=[checkpoint_period, reduce_lr, tensorboard])

    
    # if True:
    #     # 交叉熵
    #     Batch_size = 64
    #     Lr = 1e-4
    #
    #     model.compile(loss = "binary_crossentropy",
    #             optimizer = Adam(lr=Lr),
    #             metrics = ["binary_accuracy"])
    #     print('Train with batch size {}.'.format(Batch_size))
    #
    #     gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
    #     # 开始训练
    #     model.fit_generator(gen.generate(True),
    #             steps_per_epoch=max(1,train_num//Batch_size),
    #             validation_data=gen.generate(True),
    #             validation_steps=max(1,val_num//Batch_size),
    #             epochs=50,
    #             initial_epoch=31,
    #             callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])