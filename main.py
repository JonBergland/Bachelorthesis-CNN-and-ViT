import os
from resnet_trainer import ResnetTrainer

def main(dataset_root: str,
         model_name: str, 
         epochs: int = 5,
         lr_rate: float = 0.01,
         batch_size: int = 32,
         img_size: int = 128, 
         manual_seed: int = 42,
         save_path: str | None = None,
         only_see_metrics: bool = False):

    cnn = ResnetTrainer(dataset_root=dataset_root, 
                        model_name=model_name,
                        epochs = epochs,
                        lr_rate = lr_rate,
                        batch_size = batch_size,
                        img_size = img_size,
                        manual_seed = manual_seed,
                        save_path=save_path,
                        only_see_metrics=only_see_metrics)

    cnn.train()
    cnn.evaluate()

    cnn.save_model(save_optimizer=True)
    cnn.clear_model()

    cnn.plot_metrics()


    
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset")
    save_path = os.path.join(BASE_DIR, "saved_models")
    model_name = "resnet9_no_data_augmentation.pth"

    epochs = 0
    lr_rate = 0.001
    batch_size = 32
    img_size = 128
    manual_seed = 42
    only_see_metrics = False
    


    main(dataset_root=root,
         model_name=model_name,
         epochs= epochs,
         lr_rate= lr_rate,
         batch_size=batch_size,
         img_size=img_size,
         manual_seed=manual_seed,
         save_path=save_path,
         only_see_metrics=only_see_metrics)

