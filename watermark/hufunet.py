import torch
import torch.nn as nn
import os
import  tqdm
import hashlib
import numpy as np
import random
from configs.cf_watermark.cf_hufunet import epoch_attack
from networks.cnn import CnnModel
import  gc
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.utils.checkpoint import checkpoint

from util.metric import Metric
from copy import deepcopy

from torch.utils.data import DataLoader

from util.util import Random, TrainModel, Database

class HufuNet(nn.Module):
    """
    Optimal MNIST Autoencoder based on best practices and empirical results

    Architecture Details:
    - Input: 28x28x1 MNIST images
    - Latent dimension: 64 (good balance between compression and quality)
    - Uses BatchNorm for stable training
    - Uses LeakyReLU for better gradient flow
    - Progressive downsampling/upsampling
    """

    def __init__(self, latent_dim=32):
        super(HufuNet, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            # First conv block: 28x28x1 -> 14x14x16
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Second conv block: 14x14x16 -> 7x7x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14->7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Third conv block: 7x7x32 -> 4x4x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7->4 (with padding)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten and reduce to latent dimension
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),  # 1024 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),  # 128 -> 32 (latent space)
        )

        # DECODER
        self.decoder = nn.Sequential(
            # Expand from latent dimension
            nn.Linear(latent_dim, 128),  # 32 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64 * 4 * 4),  # 512 -> 2048
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to feature maps
            nn.Unflatten(1, (64, 4, 4)),  # Reshape to 128x4x4

            # First deconv block: 4x4x128 -> 7x7x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4->7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Second deconv block: 7x7x64 -> 14x14x32
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Final deconv block: 14x14x32 -> 28x28x1
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct from latent representation"""
        return self.decoder(z)

# class HufuNet(nn.Module):
#     def __init__(self, encoder_dict=None, decoder_dict=None):
#         super(HufuNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 7)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 16, 7),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )
#
#         # # Load the state dictionaries
#         if encoder_dict is not None and decoder_dict is not None:
#             self.encoder.load_state_dict(encoder_dict)
#             self.decoder.load_state_dict(decoder_dict)
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#


# def train_hufu(model, device ,train_loader, best_loss=float('inf'),epochs=30, fine_tune=False, config): -> object
def create_data_loaders_hufu(batch_size=128, validation_split=0.1):
    """
    Create optimized data loaders for MNIST
    """
    # Optimal transforms for MNIST autoencoders
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Note: No normalization for autoencoders - keep in [0,1] range
    ])

    # Load full training dataset
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Split into train/validation
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def calculate_mse(model, dataloader, device):
    model.eval()

    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, outputs = model(images)
            mse = F.mse_loss(outputs, images)
            total_mse += mse.item()
            num_batches += 1

    average_mse = total_mse / num_batches
    return average_mse

class AutoencoderMetrics:
    """
    Comprehensive metrics for autoencoder evaluation
    """

    def __init__(self, device='cpu'):
        self.device = device

    def calculate_mse(self, original, reconstructed):
        """Calculate Mean Squared Error"""
        return F.mse_loss(reconstructed, original).item()

    def calculate_mae(self, original, reconstructed):
        """Calculate Mean Absolute Error"""
        return F.l1_loss(reconstructed, original).item()

    def calculate_psnr(self, original, reconstructed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(reconstructed, original)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    # def calculate_ssim_batch(self, original, reconstructed):
    #     """Calculate Structural Similarity Index for a batch"""
    #     original_np = original.cpu().numpy()
    #     reconstructed_np = reconstructed.cpu().numpy()
    #
    #     ssim_scores = []
    #     for i in range(original_np.shape[0]):
    #         # Convert from (C, H, W) to (H, W) for grayscale
    #         orig_img = original_np[i, 0]
    #         recon_img = reconstructed_np[i, 0]
    #
    #         # Calculate SSIM
    #         ssim_score = ssim(orig_img, recon_img, data_range=1.0)
    #         ssim_scores.append(ssim_score)
    #
    #     return np.mean(ssim_scores)

    def calculate_reconstruction_accuracy(self, original, reconstructed, threshold=0.1):
        """
        Calculate reconstruction accuracy based on pixel-wise threshold
        Accuracy = percentage of pixels within threshold of original
        """
        diff = torch.abs(original - reconstructed)
        accurate_pixels = (diff <= threshold).float()
        accuracy = torch.mean(accurate_pixels).item()
        return accuracy

    def calculate_binary_accuracy(self, original, reconstructed, threshold=0.5):
        """
        Calculate binary accuracy by thresholding both images
        Useful for MNIST where pixels are mostly 0 or 1
        """
        orig_binary = (original > threshold).float()
        recon_binary = (reconstructed > threshold).float()
        accuracy = torch.mean((orig_binary == recon_binary).float()).item()
        return accuracy

def evaluate_model(model, data_loader, device, metrics_calculator):
    """
    Comprehensive model evaluation with multiple metrics
    """
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    # total_ssim = 0.0
    total_recon_acc = 0.0
    total_binary_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, reconstructed = model(data)

            # Calculate metrics
            mse = metrics_calculator.calculate_mse(data, reconstructed)
            mae = metrics_calculator.calculate_mae(data, reconstructed)
            psnr = metrics_calculator.calculate_psnr(data, reconstructed)
            # ssim_score = metrics_calculator.calculate_ssim_batch(data, reconstructed)
            recon_acc = metrics_calculator.calculate_reconstruction_accuracy(data, reconstructed)
            binary_acc = metrics_calculator.calculate_binary_accuracy(data, reconstructed)

            total_mse += mse
            total_mae += mae
            total_psnr += psnr
            # total_ssim += ssim_score
            total_recon_acc += recon_acc
            total_binary_acc += binary_acc
            num_batches += 1

    # Calculate averages
    avg_metrics = {
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches,
        'psnr': total_psnr / num_batches,
        # 'ssim': total_ssim / num_batches,
        'reconstruction_accuracy': total_recon_acc / num_batches,
        'binary_accuracy': total_binary_acc / num_batches
    }

    return avg_metrics
def train_hufu(model ,train_loader_hufu,val_loader_hufu, test_loader_hufu, config,is_fine_tune, best_loss=float('inf')) -> object:
    print("Training HufuNet")
    metrics_calculator = AutoencoderMetrics(config["device"])
    best_model = HufuNet().to(config["device"])
    if is_fine_tune==True:
        for param in model.decoder.parameters():
            param.requires_grad = False
        epochs = 15
        print("decoder freezed")
    else:
        epochs = config['epoch_hufu']
        print(f"epochs {epochs}")

    model.to(config["device"])
    criterion=nn.MSELoss()


    optimizer = torch.optim.Adam( model.parameters(), lr=config["lr_hufu"])
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm.tqdm(train_loader_hufu, leave=True)


        for batch_idx, (images, _) in enumerate(loop):
            images = images.to(config["device"])
            optimizer.zero_grad()
            _, output = model(images)
            loss = criterion(output, images)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader_hufu)
        model.eval()
        val_loss=0.0

        with torch.no_grad():
            for data, _ in val_loader_hufu:
                data = data.to(config["device"])
                _, reconstructed = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader_hufu)
        val_losses.append(avg_val_loss)
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            train_metrics = evaluate_model(model, train_loader_hufu, config["device"], metrics_calculator)
            val_metrics = evaluate_model(model, val_loader_hufu, config["device"], metrics_calculator)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'  Train - Loss: {avg_train_loss:.6f}, MSE: {train_metrics["mse"]:.6f}, '
                  f'PSNR: {train_metrics["psnr"]:.2f}'
                  # f', SSIM: {train_metrics["ssim"]:.4f}, '
                  f'Recon Acc: {train_metrics["reconstruction_accuracy"]:.4f}, '
                  f'Binary Acc: {train_metrics["binary_accuracy"]:.4f}')
            print(f'  Val   - Loss: {avg_val_loss:.6f}, MSE: {val_metrics["mse"]:.6f}, '
                  f'PSNR: {val_metrics["psnr"]:.2f},'
                  # f' SSIM: {val_metrics["ssim"]:.4f}, '
                  f'Recon Acc: {val_metrics["reconstruction_accuracy"]:.4f}, '
                  f'Binary Acc: {val_metrics["binary_accuracy"]:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')





        if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                supplementary = {'model': model,
                                 'best_loss': best_loss,
                                 'MSE':train_metrics["mse"]
                                 }
                # Save best model
                if is_fine_tune:
                    TrainModel.save_model(deepcopy(model), train_metrics["reconstruction_accuracy"],epoch, config['save_path_hufu_finetune'],
                                          supplementary)
                    best_model=deepcopy(model)
                    print(f"HufuNet model finetune saved at epoch {epoch}!")
                else:
                    TrainModel.save_model(deepcopy(model),train_metrics["reconstruction_accuracy"],  epoch, config['save_path_hufu_original'],
                                          supplementary)
                    best_model = deepcopy(model)
                    print(f"HufuNet model original saved at epoch {epoch}!")

        else:
                patience_counter += 1

        if patience_counter >= config['early_stopping_patience']:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

                # Load best model

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    final_train_metrics = evaluate_model(best_model, train_loader_hufu, config["device"], metrics_calculator)
    final_val_metrics = evaluate_model(best_model, val_loader_hufu, config["device"], metrics_calculator)
    final_test_metrics = evaluate_model(best_model, test_loader_hufu, config["device"], metrics_calculator)
    print("Final Metrics:")
    print(f"Train - MSE: {final_train_metrics['mse']:.6f}, MAE: {final_train_metrics['mae']:.6f}, "
          f"PSNR: {final_train_metrics['psnr']:.2f}dB,")
          # f" SSIM: {final_train_metrics['ssim']:.4f}")
    print(f"        Reconstruction Accuracy: {final_train_metrics['reconstruction_accuracy']:.4f}, "
          f"Binary Accuracy: {final_train_metrics['binary_accuracy']:.4f}")

    print(f"Val   - MSE: {final_val_metrics['mse']:.6f}, MAE: {final_val_metrics['mae']:.6f}, "
          f"PSNR: {final_val_metrics['psnr']:.2f}dB,")
    # SSIM: {final_val_metrics['ssim']:.4f}")
    print(f"        Reconstruction Accuracy: {final_val_metrics['reconstruction_accuracy']:.4f}, "
          f"Binary Accuracy: {final_val_metrics['binary_accuracy']:.4f}")

    print(f"Test  - MSE: {final_test_metrics['mse']:.6f}, MAE: {final_test_metrics['mae']:.6f}, "
          f"PSNR: {final_test_metrics['psnr']:.2f}dB,")
    # SSIM: {final_test_metrics['ssim']:.4f}")
    print(f"        Reconstruction Accuracy: {final_test_metrics['reconstruction_accuracy']:.4f}, "
          f"Binary Accuracy: {final_test_metrics['binary_accuracy']:.4f}")

    return model, best_val_loss#, val_losses, train_metrics_history, val_metrics_history



    #
    #     print(f"Epoch [{epoch + 1}/{config["epoch_hufu"]}], Loss: {avg_loss:.4f} , Pixel Accuracy: {avg_pixel_acc:.2f}% , MSE: {mse:.4f}")
    # return model , best_loss
def get_decoder_seed(model):
    # Extract all parameters of the decoder as a single flat tensor
    params = []
    for p in model.decoder.parameters():
        params.append(p.detach().cpu().numpy().flatten())
    flat_params = np.concatenate(params)
    # print(flat_params.size)

    # Convert parameters to a bytestring (e.g., float32 to bytes)
    byte_rep = flat_params.tobytes()
    # print(byte_rep)

    # Hash the byte string (using SHA256 for a stable, unique fingerprint)
    hash_digest = hashlib.sha256(byte_rep).hexdigest()
    # print(hash_digest)
    # Convert hash digest to an integer
    seed = int(hash_digest, 16) % (2**32)  # Use 32-bit seed
    return seed
def select_indexes(hufu, model_wat):
    seed = get_decoder_seed(hufu)
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_params_hufu = torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])
    # print(all_params_hufu.shape)

    all_params_model = torch.cat([p.data.flatten() for p in model_wat.parameters()])
    # print(all_params_model.shape)

    y = all_params_hufu.size(0)

    # Select random indices
    selected_indices = torch.randperm((all_params_model.size(0)))[:y]
    # print(y)
    # print(selected_indices.shape)
    return selected_indices

def embed_encoder_in_model(hufu, model_wat, selected_indices):
    encoder_params_hufu=torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])


    selected_indices = selected_indices.tolist()
    assert len(selected_indices) == len(encoder_params_hufu)

    current_update = 0
    count = 0

    for p in model_wat.parameters():
        numel = p.numel()
        p_flat = p.data.view(-1)

        for i in range(len(selected_indices)):
            flat_index = selected_indices[i]
            if count <= flat_index < count + numel:
                local_index = flat_index - count
                with torch.no_grad():
                    p_flat[local_index] = encoder_params_hufu[i]
        count += numel


    return  model_wat

def extract_weight_from_model(selected_indexes,model_wm,model_hufu ):
    all_params_model = torch.cat([p.data.flatten() for p in model_wm.parameters()])

    point=0

    for p in model_hufu.encoder.parameters():
            numel = p.data.numel()
            # p_flat=p.data.flatten()
            for i in range (numel):
                j=point+i
                p.data.flatten()[i]=all_params_model[selected_indexes[j]]

                # print(p.data.flatten()[i])
                # print(all_params_model[selected_indexes[j]])

            # print('hiiiiiiiiiiiiiii')
            point+=numel



    return model_hufu




# def embedding(hufu=None, initial_model=None, device=None, trainloader=None, testloader=None, epochs=50,best_loss_hufu=None,selected_indexes=None):

# def embed(hufu, initial_model, trainloader, testloader,best_loss_hufu,selected_indexes,train_loader_mnist, config)-> object:
def embed(initial_model,  testloader, trainloader,  config) -> object:
    hufu=HufuNet().to(config["device"])
    metrics_calculator2 = AutoencoderMetrics(config["device"])
    hufu_path=config["save_path_hufu_original"]
    # trainloader_mnist, testloader_mnist = Database.get_loaders(config["database_hufu"], config.get("hufu_batch_size",
    #                                                                                                config[
    #                                                                                                    "configuration"][
    #                                                                                                    "batch_size"]))

    train_loader_hufu, val_loader_hufu, test_loader_hufu = create_data_loaders_hufu(
        batch_size=config['batch_size_hufu'],
        validation_split=config['validation_split']
    )
    if os.path.exists(hufu_path):
        checkpoint=torch.load(hufu_path, config["device"],  weights_only=False)
        hufu_orig = checkpoint['supplementary']['model']  # Gets your complete model ✅
        best_loss_hufu = checkpoint['supplementary']['best_loss']  # Gets your best loss ✅
        mse_hufu = checkpoint['supplementary']['MSE']





    else:

        hufu_orig, best_loss_hufu=train_hufu(hufu ,train_loader_hufu, val_loader_hufu, test_loader_hufu, config ,is_fine_tune=False, best_loss=float('inf'))
    mse = calculate_mse(hufu, test_loader_hufu, config["device"])

    print(f"mse: {mse:.4f}")
    # print(evaluate_model(hufu,test_loader_hufu,config["device"], metrics_calculator2))


    hufu=deepcopy(hufu_orig)



    model_wat = deepcopy(initial_model).to(config["device"])



    criterion =config["criterion"]
    optimizer_cifar = torch.optim.Adam(model_wat.parameters(), lr=config["lr"])

    print("Starting training...")



    selected_indexes=select_indexes(hufu, model_wat)
    print("Selected indexes: ", selected_indexes.size())
    total = sum(p.numel() for p in hufu.encoder.parameters())
    trainable = sum(p.numel() for p in hufu.encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    total = sum(p.numel() for p in model_wat.parameters())
    trainable = sum(p.numel() for p in model_wat.parameters() if p.requires_grad)
    print(f"Total parameters model: {total:,}")
    print(f"Trainable parameters: {trainable:,}")






    best_mse=float('inf')
    best_accuracy=0
    early_stop=0
    for epoch in range(config["epochs"]):
        loss_avg = 0
        model_wat = embed_encoder_in_model(hufu, model_wat,selected_indexes)
        # test
        pp=torch.cat([p.data.flatten() for p in model_wat.parameters()])
        pp2=torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])
        my_sum=0
        for i , idx in enumerate (selected_indexes):
            # print(f"Index {idx}: {pp[idx].item()} hufu  {pp2[i]}")
            if pp[idx]==pp2[i]:
                my_sum+=1

        if my_sum==len(selected_indexes):
            print(f"epoch:{epoch+1} embeding hufu in model was ok")
        epoch_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm.tqdm(trainloader, leave=True)


        for batch_idx, (inputs, labels) in enumerate(loop):
                model_wat.train()
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                # Zero gradients
                optimizer_cifar.zero_grad()

                # Forward pass
                outputs = model_wat(inputs)

                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer_cifar.step()
                epoch_loss += loss.item()
                loop.set_description(f"watermark model training Epoch [{epoch + 1}/{config['epochs']}]")
                loop.set_postfix(
                    loss=loss.item(), avgloss=epoch_loss/(batch_idx+1),
                )
        loss_avg+=(epoch_loss/len(trainloader))

        hufu_model=extract_weight_from_model(selected_indexes,model_wat,hufu )
        pp=torch.cat([p.data.flatten() for p in model_wat.parameters()])
        pp2=torch.cat([p.data.flatten() for p in hufu_model.encoder.parameters()])
        my_sum=0
        for i , idx in enumerate (selected_indexes):
            # print(f"Index {idx}: {pp[idx].item()} hufu  {pp2[i]}")
            if pp[idx]==pp2[i]:
                my_sum+=1

        if my_sum==len(selected_indexes):
            print(f"epoch:{epoch+1} embeding model in hufu was ok")


        mse_value = calculate_mse(hufu_model, train_loader_hufu, config["device"])
        print(f"mse_value {mse_value}")

        if mse_value>0.005:
            hufu_model, best_loss_hufu=train_hufu(hufu_model ,train_loader_hufu,val_loader_hufu, test_loader_hufu,config, is_fine_tune=True, best_loss=best_loss_hufu )

        mse_value = calculate_mse(hufu_model, train_loader_hufu, config["device"])
        print(f"mse_value {mse_value}")




        with torch.no_grad():  # No gradients needed
                model_wat.eval()
                for images, labels in testloader:
                    images, labels = images.to(config["device"]), labels.to(config["device"])
                    outputs = model_wat(images)
                    _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()  # Count correct predictions
        accuracy = 100 * correct / total  # Compute accuracy percentage

        print(
                f"Epoch [{epoch + 1}/{config["epochs"]}], Test Accuracy: {accuracy:.2f}%, Classification Loss: {loss_avg:.4f}, "
                f"hufu Loss: {best_loss_hufu:.4f}, Hufu MSE: {mse_value:.4f}")
        early_stop+=1

        if accuracy>best_accuracy and mse_value<best_mse :
            early_stop=0
            best_accuracy = accuracy
            best_mse = mse_value
            supplementary = {'model': model_wat,  'watermark': hufu_model.encoder,  'selected_indexes': selected_indexes      }

            TrainModel.save_model(deepcopy(model_wat), accuracy, epoch+1, config['save_path'], supplementary)
            print("model saved!")
        elif early_stop>20:
            break







    mse_before, mse_after, mse_non_wm=extract(deepcopy(model_wat), deepcopy(hufu_model), selected_indexes, train_loader_hufu, config)
    print(f"mse_value before extraction {mse_before:.6f}")
    print(f"mse_value after extraction watermarked model {mse_after:.6f}")
    print(f"mse_value after extraction non watermarked {mse_non_wm:.6f}")
    test_hufu(hufu_model, hufu_orig)
    return model_wat
def get_model_param_norm(model, norm_type=2):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = torch.norm(param, p=norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def extract(model_watermark, hufu, selected_indexes, train_loader_hufu, config):
    mse_before=calculate_mse(hufu, train_loader_hufu, config["device"])
    norm1 = get_model_param_norm(hufu.encoder, norm_type=1)
    norm2 = get_model_param_norm(hufu.encoder, norm_type=2)
    print(f" before  norm1 {norm1:.6f} norm2 {norm2:.6f}")

    # print(f"mse_value before extraction {mse_before:.6f}")

    newhufu=extract_weight_from_model( selected_indexes, model_watermark, hufu,)
    mse_after = calculate_mse(newhufu, train_loader_hufu, config["device"])
    norm1 = get_model_param_norm(newhufu.encoder, norm_type=1)
    norm2 = get_model_param_norm(newhufu.encoder, norm_type=2)
    print(f" after  norm1 {norm1:.6f} norm2 {norm2:.6f}")
    # print(f"mse_value after extraction watermarked model {mse_after:.6f}")
    model_non = CnnModel().to(config["device"])
    checkpoint_non = torch.load("results/trained_models/cnn/_dbcifar10_ep100_bs128.pth", weights_only=False)
    model_non.load_state_dict(checkpoint_non['net'])
    hufu_non = extract_weight_from_model(selected_indexes, model_non, hufu)
    mse_non_wm = calculate_mse(hufu_non, train_loader_hufu, config["device"])
    # print(f"mse_value after extraction non watermarked {mse_non_wm:.6f}")
    return mse_before, mse_after, mse_non_wm


def test_hufu(hufu, hufu_orig):
    decoder1_params = torch.nn.utils.parameters_to_vector(hufu.decoder.parameters())
    decoder2_params = torch.nn.utils.parameters_to_vector(hufu_orig.decoder.parameters())

    difference = torch.norm(decoder1_params - decoder2_params).item()
    print(f"L2 norm of difference between decoders: {difference}")

    encoder1_params = torch.nn.utils.parameters_to_vector(hufu.encoder.parameters())
    encoder2_params = torch.nn.utils.parameters_to_vector(hufu_orig.encoder.parameters())

    difference1 = torch.norm(encoder1_params - encoder2_params).item()
    print(f"L2 norm of difference between encoders: {difference1}")


    print(decoder1_params)
    print(decoder2_params)


    print(encoder1_params)
    print(encoder2_params)


