import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import learn2learn as l2l
# Note: learn2learn.data.transforms are not explicitly used if using get_tasksets,
# but keeping imports if CustomImageDataset were to be used with manual TaskDataset setup.
# from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from tqdm import tqdm
import copy

# --- Model Definition (Consistent with reptile_ensemble_test_script) ---
class CNN4(nn.Module):
    def __init__(self, output_size, hidden_size=64, channels=3, image_size=84):
        super(CNN4, self).__init__()
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            self._conv_block(channels, hidden_size),
            self._conv_block(hidden_size, hidden_size),
            self._conv_block(hidden_size, hidden_size),
            self._conv_block(hidden_size, hidden_size)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_size, output_size)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Custom Dataset (Included for completeness, not used in this training script) ---
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        potential_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes = sorted(potential_classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.image_labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.image_labels.append(self.class_to_idx[cls_name])
        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir}. Check dataset structure and image extensions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Could not load image: {img_path}") from e
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- Utility Functions (Consistent with reptile_ensemble_test_script) ---
def accuracy_fn(predictions, targets):
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    predictions = predictions.view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def calculate_entropy(logits):
    if logits is None: 
        return float('nan') 
    probs = torch.softmax(logits, dim=1)
    clamped_probs = torch.clamp(probs, min=1e-9, max=1.0)
    if torch.isnan(clamped_probs).any():
        num_classes = logits.size(1)
        max_entropy_val = np.log(num_classes) if num_classes > 0 else 0.0
        return torch.full((logits.size(0),), float(max_entropy_val), device=logits.device).mean().item()
    try:
        entropy_per_sample = torch.distributions.Categorical(probs=clamped_probs).entropy()
    except ValueError:
        num_classes = logits.size(1)
        max_entropy_val = np.log(num_classes) if num_classes > 0 else 0.0
        entropy_per_sample = torch.full((logits.size(0),), float(max_entropy_val), device=logits.device)
    return entropy_per_sample.mean().item()

# fast_adapt for training: returns adapted learner and metrics on query set
def fast_adapt_train(batch, learner, loss_fn, adaptation_opt, adaptation_steps, 
                     support_shots, query_shots, ways, device): 
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adaptation_data_list = []
    evaluation_data_list = [] # This will be the query set for the task
    adaptation_labels_list = []
    evaluation_labels_list = []

    current_idx = 0
    for _ in range(ways):
        adaptation_data_list.append(data[current_idx : current_idx + support_shots])
        adaptation_labels_list.append(labels[current_idx : current_idx + support_shots])
        current_idx += support_shots
        evaluation_data_list.append(data[current_idx : current_idx + query_shots])
        evaluation_labels_list.append(labels[current_idx : current_idx + query_shots])
        current_idx += query_shots
    
    try:
        adaptation_data = torch.cat(adaptation_data_list, dim=0)
        adaptation_labels = torch.cat(adaptation_labels_list, dim=0)
        evaluation_data = torch.cat(evaluation_data_list, dim=0)
        evaluation_labels = torch.cat(evaluation_labels_list, dim=0)
    except Exception as e:
        print(f"Error during manual data splitting/concatenation in fast_adapt_train: {e}")
        return learner, float('nan'), float('nan'), float('nan')

    if adaptation_data.size(0) == 0:
        print(f"Warning: Empty adaptation set in fast_adapt_train.")
        if evaluation_data.size(0) == 0: 
            return learner, 0.0, 0.0, 0.0
        else: 
            with torch.no_grad():
                learner.eval() # Evaluate on query set
                eval_preds = learner(evaluation_data)
                eval_loss = loss_fn(eval_preds, evaluation_labels)
                eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
                eval_entropy = calculate_entropy(eval_preds)
            return learner, eval_loss.item(), eval_accuracy.item(), eval_entropy

    # Adaptation phase
    learner.train() 
    for step in range(adaptation_steps): 
        adaptation_opt.zero_grad()
        train_preds = learner(adaptation_data)
        train_loss = loss_fn(train_preds, adaptation_labels)
        train_loss.backward()
        adaptation_opt.step()
    
    # Evaluation on the query set of the same task
    eval_loss_item, eval_accuracy_item, eval_entropy_item = 0.0, 0.0, 0.0
    if evaluation_data.size(0) > 0:
        with torch.no_grad():
            learner.eval() # Set to eval mode for query set evaluation
            eval_preds = learner(evaluation_data)
            eval_loss = loss_fn(eval_preds, evaluation_labels)
            eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
            eval_entropy = calculate_entropy(eval_preds)
            eval_loss_item = eval_loss.item() if isinstance(eval_loss, torch.Tensor) else eval_loss
            eval_accuracy_item = eval_accuracy.item() if isinstance(eval_accuracy, torch.Tensor) else eval_accuracy
            eval_entropy_item = eval_entropy 
    else:
        print("Warning: Empty evaluation set in fast_adapt_train after adaptation.")
            
    return learner, eval_loss_item, eval_accuracy_item, eval_entropy_item

# --- Main Training Script ---
def main(args):
    base_seed = args.seed
    
    if not os.path.exists(args.save_ensemble_dir):
        os.makedirs(args.save_ensemble_dir, exist_ok=True)
        print(f"Created directory for saving ensemble models: {args.save_ensemble_dir}")

    # Define a small range for learning rate variation, e.g., +/- 10%
    lr_variation_factor = 0.1 

    for model_idx in range(args.num_ensemble_models):
        current_seed = base_seed + model_idx-2
        print(f"\n--- Training Ensemble Model {model_idx + 1}/{args.num_ensemble_models} with Seed {current_seed} ---")

        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if args.use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
        print(f"Using device: {device} for model {model_idx + 1}")

        # Calculate varied learning rates for this specific model
        # Create a simple linear variation: e.g., if 5 models, factors might be 0.9, 0.95, 1.0, 1.05, 1.1
        if args.num_ensemble_models > 1:
            # Create a perturbation factor between -1 and 1 (approximately)
            perturb_factor = (model_idx - (args.num_ensemble_models - 1) / 2.0) / (args.num_ensemble_models / 2.0)
            current_meta_lr = args.meta_lr * (1.0 + lr_variation_factor * perturb_factor)
            current_inner_lr = args.inner_lr * (1.0 + lr_variation_factor * perturb_factor)
        else: # No variation if only one model
            current_meta_lr = args.meta_lr
            current_inner_lr = args.inner_lr
        
        # Ensure LRs don't become zero or negative if base LR is very small or variation is too large
        current_meta_lr = max(1e-6, current_meta_lr) 
        current_inner_lr = max(1e-6, current_inner_lr)

        print(f"  Model {model_idx+1} using Meta LR: {current_meta_lr:.6f}, Inner LR: {current_inner_lr:.6f}")


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        print("Loading Mini-ImageNet for meta-training...")
        try:
            mini_imagenet_train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            mini_imagenet_test_transform = transforms.Compose([
                transforms.Resize([args.image_size, args.image_size]),
                transforms.ToTensor(),
                normalize
            ])
            tasksets = l2l.vision.benchmarks.get_tasksets(
                name='mini-imagenet',
                train_ways=args.ways,
                train_samples=args.shots_support + args.shots_query,
                test_ways=args.ways, 
                test_samples=args.shots_support + args.shots_query,
                root=args.data_path,
                train_transform=mini_imagenet_train_transform, 
                test_transform=mini_imagenet_test_transform, 
                download=True 
            )
            train_tasks = tasksets.train 
            print(f"Mini-ImageNet meta-training tasks loaded for model {model_idx + 1}.")
        except Exception as e:
            print(f"Error loading Mini-ImageNet for model {model_idx + 1}: {e}")
            continue 

        meta_model = CNN4(output_size=args.ways, image_size=args.image_size).to(device)
        loss_fn = nn.CrossEntropyLoss()

        print(f"Starting meta-training for model {model_idx + 1}...")
        meta_model.train()
        
        meta_train_loss_accumulator = 0.0
        meta_train_accuracy_accumulator = 0.0
        meta_train_entropy_accumulator = 0.0

        for iteration in tqdm(range(args.meta_iterations), desc=f"Model {model_idx+1} Training"):
            task_batch = train_tasks.sample() 
            
            learner = copy.deepcopy(meta_model) 
            # Use the varied inner_lr for this model's adaptation optimizer
            adaptation_opt = optim.Adam(learner.parameters(), lr=current_inner_lr) 

            adapted_learner, task_query_loss, task_query_accuracy, task_query_entropy = fast_adapt_train(
                batch=task_batch,
                learner=learner,
                loss_fn=loss_fn,
                adaptation_opt=adaptation_opt,
                adaptation_steps=args.adaptation_steps,
                support_shots=args.shots_support,
                query_shots=args.shots_query,
                ways=args.ways,                
                device=device
            )
            
            if not np.isnan(task_query_loss): meta_train_loss_accumulator += task_query_loss
            if not np.isnan(task_query_accuracy): meta_train_accuracy_accumulator += task_query_accuracy
            if not np.isnan(task_query_entropy): meta_train_entropy_accumulator += task_query_entropy
            
            # Use the varied meta_lr for this model's Reptile update
            for meta_param, learner_param in zip(meta_model.parameters(), adapted_learner.parameters()):
                if meta_param.data is not None and learner_param.data is not None:
                     meta_param.data.add_((learner_param.data - meta_param.data) * current_meta_lr)

            if (iteration + 1) % args.log_interval == 0:
                num_iters_interval = args.log_interval if args.log_interval > 0 else (iteration + 1)
                avg_meta_train_loss = meta_train_loss_accumulator / num_iters_interval
                avg_meta_train_accuracy = meta_train_accuracy_accumulator / num_iters_interval
                avg_meta_train_entropy = meta_train_entropy_accumulator / num_iters_interval
                
                print(f"\nModel {model_idx+1}, Iteration {iteration + 1}/{args.meta_iterations}: ")
                print(f"  Avg Meta-Train Loss (Query): {avg_meta_train_loss:.4f}")
                print(f"  Avg Meta-Train Accuracy (Query): {avg_meta_train_accuracy:.4f}")
                print(f"  Avg Meta-Train Entropy (Query): {avg_meta_train_entropy:.4f}")
                print(f"  Current Meta LR: {current_meta_lr:.6f}, Current Inner LR: {current_inner_lr:.6f}")
                
                meta_train_loss_accumulator = 0.0
                meta_train_accuracy_accumulator = 0.0
                meta_train_entropy_accumulator = 0.0

        print(f"Meta-training finished for model {model_idx + 1}.")
        
        model_save_name = f"reptile_ensemble_model_{model_idx}.pth"
        model_save_path = os.path.join(args.save_ensemble_dir, model_save_name)
        try:
            torch.save(meta_model.state_dict(), model_save_path)
            print(f"Meta-trained model {model_idx + 1} saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving model {model_idx + 1} to {model_save_path}: {e}")
            
    print("\n--- All Ensemble Model Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reptile Ensemble Meta-Training Script')

    parser.add_argument('--seed', type=int, default=42, help='Base random seed for ensemble training.')
    parser.add_argument('--use-cuda', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use CUDA if available')
    parser.add_argument('--data-path', type=str, default='./data/', help='Path to store/load Mini-ImageNet')
    parser.add_argument('--image-size', type=int, default=84, help='Image size (height and width)')
    parser.add_argument('--log-interval', type=int, default=100, help="Log training metrics every N iterations.")

    parser.add_argument('--num-ensemble-models', type=int, default=5, help='Number of models to train for the ensemble.')
    parser.add_argument('--save-ensemble-dir', type=str, default='./reptile_ensemble_models/', 
                        help='Directory to save the trained ensemble models.')
    # Added argument for LR variation percentage
    parser.add_argument('--lr-variation-percent', type=float, default=0.1, 
                        help='Percentage variation for LRs across ensemble models (e.g., 0.1 for +/-10%%). Set to 0 for no variation.')


    parser.add_argument('--meta-iterations', type=int, default=10000, help='Number of meta-training iterations per model.')
    parser.add_argument('--ways', type=int, default=5, help='N: Number of classes per task for meta-training')
    parser.add_argument('--shots-support', type=int, default=5, help='K_support: Support examples per class')
    parser.add_argument('--shots-query', type=int, default=15, help='K_query: Query examples per class for task evaluation') 
    parser.add_argument('--meta-lr', type=float, default=0.1, help='Base meta-learning rate for Reptile update') 
    parser.add_argument('--inner-lr', type=float, default=0.001, help='Base learning rate for inner loop adaptation')
    parser.add_argument('--adaptation-steps', type=int, default=5, help='Number of adaptation steps in the inner loop')

    args = parser.parse_args()
    
    if args.log_interval <= 0:
        print("Warning: --log-interval should be positive. Defaulting to 100.")
        args.log_interval = 100
    if args.num_ensemble_models <= 0:
        print("Error: --num-ensemble-models must be positive.")
        exit()
    if not (0 <= args.lr_variation_percent < 1.0):
        print("Warning: --lr-variation-percent should be between 0.0 and 1.0 (exclusive of 1.0). Clamping if necessary or using 0.1 if invalid.")
        if args.lr_variation_percent < 0: args.lr_variation_percent = 0.0
        if args.lr_variation_percent >=1.0: args.lr_variation_percent = 0.1 # Default variation if too high

    main(args)
