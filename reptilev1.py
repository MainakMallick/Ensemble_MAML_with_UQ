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
            raise RuntimeError(f"No images found in {root_dir}.")

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
        return torch.full((logits.size(0),), float(max_entropy_val), device=logits.device).mean().item()
    try:
        entropy_per_sample = torch.distributions.Categorical(probs=clamped_probs).entropy()
    except ValueError:
        num_classes = logits.size(1)
        max_entropy_val = np.log(num_classes) if num_classes > 0 else 0.0
        entropy_per_sample = torch.full((logits.size(0),), float(max_entropy_val), device=logits.device)
    return entropy_per_sample.mean().item()

# MODIFIED fast_adapt to use manual splitting
def fast_adapt(batch, learner, loss_fn, adaptation_opt, adaptation_steps, 
               support_shots, query_shots, ways, device, is_train=True):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Manual splitting logic
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
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, SupportShots: {support_shots}, QueryShots: {query_shots}, Ways: {ways}")
        print(f"Adaptation data list lengths: {[d.shape[0] for d in adaptation_data_list if hasattr(d, 'shape')]}")
        print(f"Evaluation data list lengths: {[d.shape[0] for d in evaluation_data_list if hasattr(d, 'shape')]}")
        return learner, float('nan'), float('nan'), float('nan')

    if adaptation_data.size(0) == 0:
        print(f"Warning: Empty adaptation set (size: {adaptation_data.size(0)}) after manual split. "
              f"Task data size: {data.size(0)}, support_shots: {support_shots}, ways: {ways}")
        if evaluation_data.size(0) == 0:
            return learner, 0.0, 0.0, 0.0 # No data to evaluate
        else: 
            with torch.no_grad():
                eval_preds = learner(evaluation_data)
                eval_loss = loss_fn(eval_preds, evaluation_labels)
                eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
                eval_entropy = calculate_entropy(eval_preds)
            return learner, eval_loss.item(), eval_accuracy.item(), eval_entropy

    if is_train: 
        learner.train() 
        for step in range(adaptation_steps):
            adaptation_opt.zero_grad()
            train_preds = learner(adaptation_data)
            train_loss = loss_fn(train_preds, adaptation_labels)
            train_loss.backward()
            adaptation_opt.step()
    else: 
        learner.train() # Still train during adaptation phase for test, then eval
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
            print("Warning: Empty evaluation set in fast_adapt. Returning 0 for metrics.")
            eval_loss_item, eval_accuracy_item, eval_entropy_item = 0.0, 0.0, 0.0
            
    return learner, eval_loss_item, eval_accuracy_item, eval_entropy_item

# --- Main Script ---
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")

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
            test_ways=args.ways_test, 
            test_samples=args.shots_support_test + args.shots_query_test,
            root=args.data_path,
            train_transform=mini_imagenet_train_transform, 
            test_transform=mini_imagenet_test_transform,
            download=True
        )
        train_tasks = tasksets.train 
        print(f"Mini-ImageNet meta-training tasks loaded.")
    except Exception as e:
        print(f"Error loading Mini-ImageNet: {e}")
        return

    print(f"Loading custom dataset from: {args.custom_data_path} for meta-testing...")
    if args.custom_data_path and os.path.exists(args.custom_data_path):
        custom_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]), 
            transforms.ToTensor(),
            normalize 
        ])
        try:
            test_custom_dataset_raw = CustomImageDataset(root_dir=args.custom_data_path, transform=custom_transform)
            if len(test_custom_dataset_raw.classes) < args.ways_test:
                raise ValueError(f"Custom dataset has {len(test_custom_dataset_raw.classes)} classes, "
                                 f"but {args.ways_test}-way evaluation requested.")
            
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
            print(f"Custom meta-testing tasks loaded. Num tasks: {len(test_tasks_custom)}")
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            test_tasks_custom = None
    else:
        print("Custom dataset path not provided or invalid. Skipping meta-testing.")
        test_tasks_custom = None

    meta_model = CNN4(output_size=args.ways, image_size=args.image_size).to(device)
    loss_fn = nn.CrossEntropyLoss()

    print("\nStarting meta-training...")
    meta_model.train()
    
    # MODIFIED: Add accumulators for meta-training metrics
    meta_train_loss_accumulator = 0.0
    meta_train_accuracy_accumulator = 0.0
    meta_train_entropy_accumulator = 0.0

    for iteration in tqdm(range(args.meta_iterations), desc="Meta-Training Iterations"):
        task_batch = train_tasks.sample() 
        
        learner = copy.deepcopy(meta_model) 
        adaptation_opt = optim.Adam(learner.parameters(), lr=args.inner_lr)

        # MODIFIED: Capture the metrics from fast_adapt
        adapted_learner, task_query_loss, task_query_accuracy, task_query_entropy = fast_adapt(
            batch=task_batch,
            learner=learner,
            loss_fn=loss_fn,
            adaptation_opt=adaptation_opt,
            adaptation_steps=args.adaptation_steps,
            support_shots=args.shots_support,
            query_shots=args.shots_query,
            ways=args.ways,                
            device=device,
            is_train=True
        )
        
        # MODIFIED: Accumulate metrics, handling potential NaNs
        if not np.isnan(task_query_loss):
            meta_train_loss_accumulator += task_query_loss
        if not np.isnan(task_query_accuracy):
            meta_train_accuracy_accumulator += task_query_accuracy
        if not np.isnan(task_query_entropy):
            meta_train_entropy_accumulator += task_query_entropy
        
        # Reptile meta-update (remains the same)
        for meta_param, learner_param in zip(meta_model.parameters(), adapted_learner.parameters()):
            # Ensure meta_param.data and learner_param.data are not None
            if meta_param.data is not None and learner_param.data is not None:
                 meta_param.data.add_((learner_param.data - meta_param.data) * args.meta_lr)
            elif meta_param.grad is not None and learner_param.grad is not None: # Fallback for some specific cases if data is None but grad is used
                 meta_param.grad.add_((learner_param.grad - meta_param.grad) * args.meta_lr)


        # MODIFIED: Log accumulated metrics at specified intervals
        if (iteration + 1) % args.log_interval == 0:
            # Avoid division by zero if log_interval is 0 or somehow no iterations ran yet (though tqdm starts from 0)
            # Ensure args.log_interval is positive before division
            num_iterations_in_interval = args.log_interval if args.log_interval > 0 else (iteration + 1)

            avg_meta_train_loss = meta_train_loss_accumulator / num_iterations_in_interval
            avg_meta_train_accuracy = meta_train_accuracy_accumulator / num_iterations_in_interval
            avg_meta_train_entropy = meta_train_entropy_accumulator / num_iterations_in_interval
            
            print(f"\nIteration {iteration + 1}/{args.meta_iterations}: ")
            print(f"  Avg Meta-Train Loss (Query): {avg_meta_train_loss:.4f}")
            print(f"  Avg Meta-Train Accuracy (Query): {avg_meta_train_accuracy:.4f}")
            print(f"  Avg Meta-Train Entropy (Query): {avg_meta_train_entropy:.4f}")
            print(f"  Meta LR: {args.meta_lr}")
            
            # Reset accumulators for the next logging interval
            meta_train_loss_accumulator = 0.0
            meta_train_accuracy_accumulator = 0.0
            meta_train_entropy_accumulator = 0.0

    print("Meta-training finished.")
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        torch.save(meta_model.state_dict(), args.save_path)
        print(f"Meta-trained model saved to {args.save_path}")

    if test_tasks_custom:
        print("\nStarting meta-testing on custom dataset...")
        meta_model.eval() 

        total_test_accuracy = 0.0
        total_test_entropy = 0.0
        successful_test_tasks = 0
        
        for test_iter in tqdm(range(args.meta_test_tasks), desc="Meta-Testing Tasks"):
            task_batch = test_tasks_custom.sample()
            
            learner = copy.deepcopy(meta_model) 
            
            if args.ways != args.ways_test: # Adjust classifier for different ways in testing
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
                is_train=False 
            )
            
            if not (np.isnan(task_accuracy) or np.isnan(task_entropy)):
                total_test_accuracy += task_accuracy
                total_test_entropy += task_entropy
                successful_test_tasks += 1
            else:
                print(f"Warning: Task {test_iter+1} resulted in NaN metrics. Skipping.")

            # Log first few tasks and then periodically
            if test_iter < 5 or (test_iter + 1) % (max(1, args.meta_test_tasks // 10)) == 0 :
                print(f"Test Task {test_iter+1}/{args.meta_test_tasks}: Acc = {task_accuracy:.4f}, Entropy = {task_entropy:.4f}")

        if successful_test_tasks > 0:
            avg_test_accuracy = total_test_accuracy / successful_test_tasks
            avg_test_entropy = total_test_entropy / successful_test_tasks
        else:
            avg_test_accuracy = 0.0
            avg_test_entropy = 0.0
            if args.meta_test_tasks > 0 :
                 print("Warning: All test tasks resulted in NaN or no successful tasks to report metrics for meta-testing.")

        print("\n--- Meta-Testing Results on Custom Dataset ---")
        print(f"Average Few-Shot Accuracy (over {successful_test_tasks} successful tasks): {avg_test_accuracy:.4f}")
        print(f"Average Predictive Entropy (over {successful_test_tasks} successful tasks): {avg_test_entropy:.4f}")
    else:
        print("Skipping meta-testing as no valid custom dataset was provided or loaded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reptile Meta-Learning with Mini-ImageNet and Custom Dataset')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cuda', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--data-path', type=str, default='C:/Users/mmallick7/Downloads/learn2learn-master/data/')
    parser.add_argument('--custom-data-path', type=str, 
                        default='C:/Users/mmallick7/Downloads/learn2learn-master/learn2learn-master/Dataset/test', 
                        help='Path to your custom image dataset for meta-testing. Defaults to the hardcoded path.')
    parser.add_argument('--save-path', type=str, default='C:/Users/mmallick7/Downloads/learn2learn-master/data/reptile_meta_model.pth')
    parser.add_argument('--log-interval', type=int, default=100, help="Log training metrics every N iterations.")
    parser.add_argument('--image-size', type=int, default=84)

    parser.add_argument('--meta-iterations', type=int, default=10000)
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots-support', type=int, default=5)
    parser.add_argument('--shots-query', type=int, default=15) 
    parser.add_argument('--meta-lr', type=float, default=0.1) 
    parser.add_argument('--inner-lr', type=float, default=0.001)
    parser.add_argument('--adaptation-steps', type=int, default=5)

    parser.add_argument('--meta-test-tasks', type=int, default=100)
    parser.add_argument('--ways-test', type=int, default=4)
    parser.add_argument('--shots-support-test', type=int, default=5)
    parser.add_argument('--shots-query-test', type=int, default=15)
    parser.add_argument('--inner-lr-test', type=float, default=0.001)
    parser.add_argument('--adaptation-steps-test', type=int, default=10)

    args = parser.parse_args()
    if args.log_interval <= 0:
        print("Warning: --log-interval should be positive. Defaulting to 100 if it was set to 0 or less for safety.")
        args.log_interval = 100 # Ensure it's positive

    main(args)