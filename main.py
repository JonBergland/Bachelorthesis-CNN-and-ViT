import os
from resnet_trainer import ResnetTrainer
from vision_transformer_trainer import VisionTransformerTrainer

def main(dataset_root: str,
         model_name: str, 
         epochs: int = 5,
         lr_rate: float = 0.01,
         batch_size: int = 32,
         img_size: int = 128, 
         manual_seed: int = 42,
         save_path: str | None = None,
         only_see_metrics: bool = False,
         dropout_rate: float = 0.6,
         label_smoothing: float = 0.05,
         weight_decay: float = 2e-3,
         lr_step_size: int = 5,
         lr_gamma: float = 0.2):

    # cnn = ResnetTrainer(dataset_root=dataset_root, 
    #                     model_name=model_name,
    #                     epochs = epochs,
    #                     lr_rate = lr_rate,
    #                     batch_size = batch_size,
    #                     img_size = img_size,
    #                     manual_seed = manual_seed,
    #                     save_path=save_path,
    #                     only_see_metrics=only_see_metrics,
    #                     dropout_rate=dropout_rate,
    #                     label_smoothing=label_smoothing,
    #                     weight_decay=weight_decay,
    #                     lr_step_size=lr_step_size,
    #                     lr_gamma=lr_gamma)

    

    # cnn.train()
    # cnn.evaluate()

    # cnn.save_model(model=cnn.model, save_optimizer=True)
    # cnn.clear_model()

    # cnn.plot_metrics()

    vit = VisionTransformerTrainer(dataset_root=dataset_root, 
                        model_name=model_name,
                        epochs = epochs,
                        lr_rate = lr_rate,
                        batch_size = batch_size,
                        img_size = img_size,
                        manual_seed = manual_seed,
                        save_path=save_path,
                        only_see_metrics=only_see_metrics,
                        dropout_rate=dropout_rate,
                        label_smoothing=label_smoothing,
                        weight_decay=weight_decay,
                        lr_step_size=lr_step_size,
                        lr_gamma=lr_gamma)
    
    
    vit.train()
    vit.evaluate()

    vit.save_model(model=vit.model, save_optimizer=True)
    vit.clear_model()

    vit.plot_metrics()

    
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset")
    save_path = os.path.join(BASE_DIR, "saved_models")
    model_name = "vit_no_data_augmentation.pth"

    epochs = 1
    lr_rate = 1e-4
    batch_size = 32
    img_size = 128
    manual_seed = 42
    only_see_metrics = False

    
    # Overfitting measures
    dropout_rate = 0.6
    label_smoothing = 0.05
    weight_decay = 2e-3

    # Learning rate
    lr_step_size = 5
    lr_gamma = 0.2

    


    main(dataset_root=root,
         model_name=model_name,
         epochs= epochs,
         lr_rate= lr_rate,
         batch_size=batch_size,
         img_size=img_size,
         manual_seed=manual_seed,
         save_path=save_path,
            only_see_metrics=only_see_metrics,
            dropout_rate=dropout_rate,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma)

