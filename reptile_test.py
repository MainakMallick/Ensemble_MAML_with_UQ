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
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from tqdm import tqdm
import copy

# --- Model Definition ---
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

# --- Custom Dataset ---
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

# --- Utility Functions ---
def accuracy_fn(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    clamped_probs = torch.clamp(probs, min=1e-9, max=1.0)
    if torch.isnan(clamped_probs).any():
        num_classes = logits.size(1)
        max_entropy_val = np.log(num_classes) if num_classes > 0 else 0.0
        # Return as a scalar item
        return torch.full((logits.size(0),), float(max_entropy_val), device=logits.device).mean().item()
    try:
        entropy_per_sample = torch.distributions.Categorical(probs=clamped_probs).entropy()
    except ValueError:
        num_classes = logits.size(1)
        max_entropy_val = np.log(num_classes) if num_classes > 0 else 0.0
        entropy_per_sample = torch.full((logits.size(0),), float(max_entropy_val), device=logits.device)
    return entropy_per_sample.mean().item() # Return as a scalar item

def fast_adapt(batch, learner, loss_fn, adaptation_opt, adaptation_steps, 
               support_shots, query_shots, ways, device, is_train=True): # is_train is less relevant here but kept for signature
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adaptation_data_list = []
    evaluation_data_list = []
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
        print(f"Error during manual data splitting/concatenation: {e}")
        return learner, float('nan'), float('nan'), float('nan')

    if adaptation_data.size(0) == 0:
        print(f"Warning: Empty adaptation set after manual split.")
        if evaluation_data.size(0) == 0: return learner, 0.0, 0.0, 0.0
        else: 
            with torch.no_grad():
                eval_preds = learner(evaluation_data)
                eval_loss = loss_fn(eval_preds, evaluation_labels)
                eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
                eval_entropy = calculate_entropy(eval_preds)
            return learner, eval_loss.item(), eval_accuracy.item(), eval_entropy

    # For testing, we still adapt the learner
    learner.train() # Enable train mode for adaptation (e.g., for BatchNorm updates)
    for step in range(adaptation_steps): 
        adaptation_opt.zero_grad()
        train_preds = learner(adaptation_data)
        train_loss = loss_fn(train_preds, adaptation_labels)
        train_loss.backward()
        adaptation_opt.step()
    learner.eval() # Set to eval mode for final evaluation on query set

    with torch.no_grad():
        if evaluation_data.size(0) > 0:
            eval_preds = learner(evaluation_data)
            eval_loss = loss_fn(eval_preds, evaluation_labels)
            eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
            eval_entropy = calculate_entropy(eval_preds)
            eval_loss_item = eval_loss.item() if isinstance(eval_loss, torch.Tensor) else eval_loss
            eval_accuracy_item = eval_accuracy.item() if isinstance(eval_accuracy, torch.Tensor) else eval_accuracy
            eval_entropy_item = eval_entropy
        else:
            eval_loss_item, eval_accuracy_item, eval_entropy_item = 0.0, 0.0, 0.0
            
    return learner, eval_loss_item, eval_accuracy_item, eval_entropy_item

# --- Main Testing Script ---
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # --- Load Custom Dataset for Meta-Testing ---
    test_tasks_custom = None
    if args.custom_data_path and os.path.exists(args.custom_data_path) and os.path.isdir(args.custom_data_path):
        print(f"Loading custom dataset from: {args.custom_data_path}")
        custom_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]), 
            transforms.ToTensor(),
            normalize 
        ])
        try:
            test_custom_dataset_raw = CustomImageDataset(root_dir=args.custom_data_path, transform=custom_transform)
            
            if not test_custom_dataset_raw.image_labels:
                print(f"ERROR: No images (and thus no labels) were loaded from '{args.custom_data_path}'.")
            else:
                unique_loaded_labels = torch.unique(torch.tensor(test_custom_dataset_raw.image_labels)).tolist()
                num_classes_with_images = len(unique_loaded_labels)
                print(f"DEBUG: Custom dataset has {num_classes_with_images} classes with actual images.")

                if num_classes_with_images < args.ways_test:
                    print(f"ERROR: Custom dataset has only {num_classes_with_images} classes with images, "
                          f"but {args.ways_test}-way evaluation is requested. Need at least {args.ways_test} classes.")
                else:
                    meta_test_custom_dataset = l2l.data.MetaDataset(test_custom_dataset_raw)
                    test_task_transforms = [
                        NWays(meta_test_custom_dataset, n=args.ways_test),
                        KShots(meta_test_custom_dataset, k=args.shots_support_test + args.shots_query_test), 
                        LoadData(meta_test_custom_dataset),
                        RemapLabels(meta_test_custom_dataset), 
                        ConsecutiveLabels(meta_test_custom_dataset) 
                    ]
                    test_tasks_custom = l2l.data.TaskDataset(meta_test_custom_dataset,
                                                             task_transforms=test_task_transforms,
                                                             num_tasks=args.meta_test_tasks)
                    print(f"Custom meta-testing tasks successfully created. Num tasks: {len(test_tasks_custom)}")
        except Exception as e:
            print(f"ERROR: Could not load or process custom dataset from '{args.custom_data_path}': {e}")
            import traceback
            traceback.print_exc()
    else:
        if not args.custom_data_path: print("ERROR: Custom dataset path not provided (--custom-data-path).")
        elif not os.path.exists(args.custom_data_path): print(f"ERROR: Custom dataset path does not exist: '{args.custom_data_path}'.")
        else: print(f"ERROR: Custom dataset path is not a directory: '{args.custom_data_path}'.")

    if not test_tasks_custom:
        print("Exiting: Cannot proceed with testing without a valid custom dataset.")
        return

    # --- Load Meta-Trained Model ---
    # Initialize model structure. args.ways should match the original training 'ways' for the saved model's head.
    meta_model = CNN4(output_size=args.ways_train, image_size=args.image_size).to(device)
    loss_fn = nn.CrossEntropyLoss() # Define loss_fn as it's needed for fast_adapt

    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"Loading pre-trained model from {args.load_model_path}...")
        try:
            meta_model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {args.load_model_path}: {e}")
            print("Ensure --ways-train argument matches the training configuration of the saved model if issues persist.")
            return
    else:
        if not args.load_model_path: print("ERROR: Path to saved model not provided (--load-model-path).")
        else: print(f"ERROR: Saved model path does not exist: '{args.load_model_path}'.")
        print("Exiting: Cannot proceed without a model to test.")
        return

    # --- Meta-Testing on Custom Dataset ---
    print("\nStarting meta-testing with loaded model...")
    meta_model.eval() 

    total_test_accuracy = 0.0
    total_test_entropy = 0.0
    successful_test_tasks = 0
    
    for test_iter in tqdm(range(args.meta_test_tasks), desc="Meta-Testing Tasks"):
        task_batch = test_tasks_custom.sample()
        
        learner = copy.deepcopy(meta_model) 
        
        # Adjust classifier head if test ways differ from the (potentially loaded) meta_model's original training ways
        if args.ways_train != args.ways_test:
            num_features = learner.classifier.in_features 
            learner.classifier = nn.Linear(num_features, args.ways_test).to(device)
            nn.init.kaiming_normal_(learner.classifier.weight, mode='fan_out', nonlinearity='relu')
            if learner.classifier.bias is not None:
                learner.classifier.bias.data.zero_()

        test_adaptation_opt = optim.Adam(learner.parameters(), lr=args.inner_lr_test)

        _, _, task_accuracy, task_entropy = fast_adapt(
            batch=task_batch,
            learner=learner,
            loss_fn=loss_fn,
            adaptation_opt=test_adaptation_opt, 
            adaptation_steps=args.adaptation_steps_test,
            support_shots=args.shots_support_test,
            query_shots=args.shots_query_test,
            ways=args.ways_test,
            device=device,
            is_train=False # Kept for signature, adaptation still happens
        )
        
        if not (np.isnan(task_accuracy) or np.isnan(task_entropy)):
            total_test_accuracy += task_accuracy
            total_test_entropy += task_entropy
            successful_test_tasks += 1
        else:
            print(f"Warning: Task {test_iter+1} resulted in NaN metrics. Skipping.")

        if test_iter < 5 or (test_iter + 1) % (max(1, args.meta_test_tasks // 10)) == 0 :
            print(f"Test Task {test_iter+1}/{args.meta_test_tasks}: Acc = {task_accuracy:.4f}, Entropy = {task_entropy:.4f}")

    if successful_test_tasks > 0:
        avg_test_accuracy = total_test_accuracy / successful_test_tasks
        avg_test_entropy = total_test_entropy / successful_test_tasks
    else:
        avg_test_accuracy = 0.0; avg_test_entropy = 0.0
        if args.meta_test_tasks > 0 :
             print("Warning: All test tasks resulted in NaN or no successful tasks to report metrics for meta-testing.")

    print("\n--- Meta-Testing Results on Custom Dataset ---")
    print(f"Average Few-Shot Accuracy (over {successful_test_tasks} successful tasks): {avg_test_accuracy:.4f}")
    print(f"Average Predictive Entropy (over {successful_test_tasks} successful tasks): {avg_test_entropy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reptile Meta-Learning - TEST ONLY SCRIPT')

    # General args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-cuda', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use CUDA if available')
    parser.add_argument('--image-size', type=int, default=84, help='Image size (height and width)')
    
    # Model and Data Paths
    parser.add_argument('--load-model-path', type=str, default='C:/Users/mmallick7/Downloads/learn2learn-master/data/reptile_meta_model.pth', help='Path to the saved meta-trained model (.pth file)')
    parser.add_argument('--custom-data-path', type=str, 
                        default='C:/Users/mmallick7/Downloads/learn2learn-master/learn2learn-master/Dataset/test',
                        help='Path to your custom image dataset for meta-testing.')
    
    # Meta-Model Structure (should match the loaded model's training config)
    parser.add_argument('--ways-train', type=int, default=5, 
                        help='N-ways the loaded meta-model was originally trained with (for initializing model head correctly before loading state_dict).')

    # Meta-Testing args
    parser.add_argument('--meta-test-tasks', type=int, default=100, help='Number of tasks for meta-testing')
    parser.add_argument('--ways-test', type=int, default=4, help='N_test: Number of classes per task for meta-testing')
    parser.add_argument('--shots-support-test', type=int, default=5, help='K_support_test: Number of support examples per class')
    parser.add_argument('--shots-query-test', type=int, default=15, help='K_query_test: Number of query examples per class')
    parser.add_argument('--inner-lr-test', type=float, default=0.001, help='Learning rate for inner loop adaptation during testing')
    parser.add_argument('--adaptation-steps-test', type=int, default=100, help='Number of adaptation steps during testing')

    args = parser.parse_args()
    main(args)
