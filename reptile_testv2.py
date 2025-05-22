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
def accuracy_fn(predictions, targets): # Can take logits or probabilities
    if predictions.ndim == 2 and predictions.shape[1] > 1: # Logits or probabilities
        predictions = predictions.argmax(dim=1)
    predictions = predictions.view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def calculate_entropy(logits):
    if logits is None: 
        print("Warning: calculate_entropy received None logits.")
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

def fast_adapt(batch, learner, loss_fn, adaptation_opt, adaptation_steps, 
               support_shots, query_shots, ways, device, is_train_flag=True): 
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
        print(f"Error during manual data splitting/concatenation in fast_adapt: {e}")
        return learner, float('nan'), float('nan'), float('nan'), None 

    if adaptation_data.size(0) == 0:
        print(f"Warning: Empty adaptation set in fast_adapt.")
        eval_preds_to_return = None
        if evaluation_data.size(0) == 0: 
            return learner, 0.0, 0.0, 0.0, None
        else: 
            with torch.no_grad():
                eval_preds = learner(evaluation_data)
                eval_preds_to_return = eval_preds
                eval_loss = loss_fn(eval_preds, evaluation_labels)
                eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
                eval_entropy = calculate_entropy(eval_preds)
            return learner, eval_loss.item(), eval_accuracy.item(), eval_entropy, eval_preds_to_return

    learner.train() 
    for step in range(adaptation_steps): 
        adaptation_opt.zero_grad()
        train_preds = learner(adaptation_data)
        train_loss = loss_fn(train_preds, adaptation_labels)
        train_loss.backward()
        adaptation_opt.step()
    learner.eval() 

    eval_preds_to_return = None
    eval_loss_item, eval_accuracy_item, eval_entropy_item = 0.0, 0.0, 0.0
    with torch.no_grad():
        if evaluation_data.size(0) > 0:
            eval_preds = learner(evaluation_data)
            eval_preds_to_return = eval_preds 
            eval_loss = loss_fn(eval_preds, evaluation_labels)
            eval_accuracy = accuracy_fn(eval_preds, evaluation_labels)
            eval_entropy = calculate_entropy(eval_preds)
            eval_loss_item = eval_loss.item() if isinstance(eval_loss, torch.Tensor) else eval_loss
            eval_accuracy_item = eval_accuracy.item() if isinstance(eval_accuracy, torch.Tensor) else eval_accuracy
            eval_entropy_item = eval_entropy 
        else:
            print("Warning: Empty evaluation set in fast_adapt after adaptation.")
            
    return learner, eval_loss_item, eval_accuracy_item, eval_entropy_item, eval_preds_to_return

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
                print(f"ERROR: No images loaded from '{args.custom_data_path}'.")
            else:
                unique_loaded_labels = torch.unique(torch.tensor(test_custom_dataset_raw.image_labels)).tolist()
                num_classes_with_images = len(unique_loaded_labels)
                print(f"DEBUG: Custom dataset has {num_classes_with_images} classes with actual images.")
                if num_classes_with_images < args.ways_test:
                    print(f"ERROR: Need at least {args.ways_test} classes with images for {args.ways_test}-way testing. Found {num_classes_with_images}.")
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
                    print(f"Custom meta-testing tasks created. Num tasks: {len(test_tasks_custom)}")
        except Exception as e:
            print(f"ERROR loading custom dataset: {e}")
            import traceback; traceback.print_exc()
    else: 
        if not args.custom_data_path: print("ERROR: Custom dataset path not provided.")
        elif not os.path.exists(args.custom_data_path): print(f"ERROR: Path does not exist: '{args.custom_data_path}'.")
        else: print(f"ERROR: Path is not a directory: '{args.custom_data_path}'.")

    if not test_tasks_custom:
        print("Exiting: Cannot test without a valid custom dataset."); return

    ensemble_meta_models = []
    model_file_paths_to_load = []

    if args.load_models_dir:
        if os.path.exists(args.load_models_dir) and os.path.isdir(args.load_models_dir):
            print(f"Scanning directory for models: {args.load_models_dir}")
            for filename in os.listdir(args.load_models_dir):
                if filename.lower().endswith(".pth"):
                    model_file_paths_to_load.append(os.path.join(args.load_models_dir, filename))
            if not model_file_paths_to_load:
                print(f"ERROR: No '.pth' model files found in directory: {args.load_models_dir}. Exiting.")
                return
        else:
            print(f"ERROR: Provided model directory does not exist or is not a directory: {args.load_models_dir}. Exiting.")
            return
    else:
        print("ERROR: No model directory provided for the ensemble (--load-models-dir). Exiting.")
        return

    print(f"Attempting to load {len(model_file_paths_to_load)} model(s) for the ensemble...")
    for model_idx, model_path in enumerate(model_file_paths_to_load):
        # No need to check os.path.exists again, already filtered
        print(f"Loading model {model_idx+1}/{len(model_file_paths_to_load)} from {model_path}...")
        current_meta_model = CNN4(output_size=args.ways_train, image_size=args.image_size).to(device)
        try:
            current_meta_model.load_state_dict(torch.load(model_path, map_location=device))
            current_meta_model.eval() 
            ensemble_meta_models.append(current_meta_model)
            print(f"Model {model_idx+1} loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}. Ensure --ways-train matches its training.")
    
    if not ensemble_meta_models:
        print("ERROR: No models were successfully loaded for the ensemble from the specified directory. Exiting."); return
    print(f"Successfully loaded {len(ensemble_meta_models)} models into the ensemble.")
    
    loss_fn = nn.CrossEntropyLoss()

    print("\nStarting meta-testing with ensemble...")

    total_ensemble_accuracy = 0.0
    total_ensemble_entropy_of_mean = 0.0 
    total_avg_individual_entropy = 0.0   
    total_disagreement = 0.0             
    successful_test_tasks = 0
    
    for test_iter in tqdm(range(args.meta_test_tasks), desc="Meta-Testing Tasks"):
        task_batch = test_tasks_custom.sample()
        
        temp_data, temp_labels = task_batch
        temp_data, temp_labels = temp_data.to(device), temp_labels.to(device)
        
        _eval_labels_list_main = []
        _current_idx_main = 0
        for _ in range(args.ways_test):
            _current_idx_main += args.shots_support_test
            _eval_labels_list_main.append(temp_labels[_current_idx_main : _current_idx_main + args.shots_query_test])
            _current_idx_main += args.shots_query_test
        try:
            evaluation_labels_for_task = torch.cat(_eval_labels_list_main, dim=0)
            if evaluation_labels_for_task.size(0) == 0 and args.shots_query_test > 0 :
                 print(f"Warning: Task {test_iter+1} resulted in zero query labels. Skipping.")
                 continue
        except Exception as e:
            print(f"Error splitting labels for task {test_iter+1}: {e}. Skipping task.")
            continue

        all_query_logits_for_task = []
        individual_entropies_for_task = [] 

        for model_idx, meta_model_i in enumerate(ensemble_meta_models):
            learner_i = copy.deepcopy(meta_model_i)
            
            if args.ways_train != args.ways_test:
                num_features = learner_i.classifier.in_features 
                learner_i.classifier = nn.Linear(num_features, args.ways_test).to(device)
                nn.init.kaiming_normal_(learner_i.classifier.weight, mode='fan_out', nonlinearity='relu')
                if learner_i.classifier.bias is not None:
                    learner_i.classifier.bias.data.zero_()

            test_adaptation_opt_i = optim.Adam(learner_i.parameters(), lr=args.inner_lr_test)

            _, _, _, individual_entropy_i, query_logits_i = fast_adapt(
                batch=task_batch, 
                learner=learner_i,
                loss_fn=loss_fn,
                adaptation_opt=test_adaptation_opt_i,
                adaptation_steps=args.adaptation_steps_test,
                support_shots=args.shots_support_test,
                query_shots=args.shots_query_test,
                ways=args.ways_test,
                device=device,
                is_train_flag=False 
            )
            
            if query_logits_i is not None and not (isinstance(query_logits_i, float) and np.isnan(query_logits_i)):
                if query_logits_i.shape[0] != evaluation_labels_for_task.shape[0]:
                    print(f"Warning: Mismatch in query logits and labels for model {model_idx+1}, task {test_iter+1}. Skipping member.")
                    continue 
                all_query_logits_for_task.append(query_logits_i)
                if not np.isnan(individual_entropy_i): 
                    individual_entropies_for_task.append(individual_entropy_i)
            else:
                print(f"Warning: fast_adapt for model {model_idx+1}, task {test_iter+1} returned invalid logits. Skipping member.")
        
        if not all_query_logits_for_task: 
            print(f"Warning: No ensemble members produced valid logits for task {test_iter+1}. Skipping task.")
            continue 

        stacked_query_logits = torch.stack(all_query_logits_for_task, dim=0) 
        ensemble_mean_logits = torch.mean(stacked_query_logits, dim=0)      
        
        task_ensemble_accuracy = accuracy_fn(ensemble_mean_logits, evaluation_labels_for_task)
        task_entropy_of_mean = calculate_entropy(ensemble_mean_logits)
        
        task_avg_individual_entropy = np.mean(individual_entropies_for_task) if individual_entropies_for_task else float('nan')

        stacked_query_probs = torch.softmax(stacked_query_logits, dim=-1) 
        variance_of_probs = torch.var(stacked_query_probs, dim=0) 
        task_disagreement = variance_of_probs.mean().item() if variance_of_probs.numel() > 0 else float('nan')

        if not (np.isnan(task_ensemble_accuracy.item()) or 
                np.isnan(task_entropy_of_mean) or 
                np.isnan(task_avg_individual_entropy) or 
                np.isnan(task_disagreement)):
            total_ensemble_accuracy += task_ensemble_accuracy.item() 
            total_ensemble_entropy_of_mean += task_entropy_of_mean 
            total_avg_individual_entropy += task_avg_individual_entropy
            total_disagreement += task_disagreement
            successful_test_tasks += 1
        else:
            print(f"Warning: Task {test_iter+1} (ensemble) resulted in one or more NaN metrics. Skipping accumulation.")

        if test_iter < 5 or (test_iter + 1) % (max(1, args.meta_test_tasks // 10)) == 0 :
            print(f"Test Task {test_iter+1}/{args.meta_test_tasks}: Ens Acc = {task_ensemble_accuracy.item():.4f}, "
                  f"Ens Entropy = {task_entropy_of_mean:.4f}, "
                  f"Avg Indiv Entropy = {task_avg_individual_entropy:.4f}, "
                  f"Disagreement = {task_disagreement:.4f}")

    if successful_test_tasks > 0:
        avg_ensemble_accuracy = total_ensemble_accuracy / successful_test_tasks
        avg_ensemble_entropy_of_mean = total_ensemble_entropy_of_mean / successful_test_tasks
        avg_avg_individual_entropy = total_avg_individual_entropy / successful_test_tasks
        avg_disagreement = total_disagreement / successful_test_tasks
    else:
        avg_ensemble_accuracy = avg_ensemble_entropy_of_mean = avg_avg_individual_entropy = avg_disagreement = 0.0
        if args.meta_test_tasks > 0 :
             print("Warning: All test tasks resulted in NaN or no successful tasks for ensemble.")

    print("\n--- Ensemble Meta-Testing Results on Custom Dataset ---")
    print(f"Number of models in ensemble: {len(ensemble_meta_models)}")
    print(f"Number of successful test tasks: {successful_test_tasks}/{args.meta_test_tasks}")
    print(f"Average Ensemble Few-Shot Accuracy: {avg_ensemble_accuracy:.4f}")
    print(f"Average Predictive Entropy of Ensemble Mean: {avg_ensemble_entropy_of_mean:.4f}")
    print(f"Average of Individual Model Entropies: {avg_avg_individual_entropy:.4f}")
    print(f"Average Disagreement (Predictive Variance): {avg_disagreement:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reptile Ensemble Meta-Learning - TEST ONLY SCRIPT')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cuda', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--image-size', type=int, default=84)
    
    # MODIFIED: Takes a directory path now
    parser.add_argument('--load-models-dir', type=str, required=True, 
                        help='Directory path containing saved meta-trained models (.pth files) for the ensemble.')
    parser.add_argument('--custom-data-path', type=str, 
                        default='C:/Users/mmallick7/Downloads/learn2learn-master/learn2learn-master/Dataset/test', 
                        help='Path to custom image dataset for meta-testing.')
    
    parser.add_argument('--ways-train', type=int, default=5, 
                        help='N-ways the loaded models were originally trained with.')

    parser.add_argument('--meta-test-tasks', type=int, default=100)
    parser.add_argument('--ways-test', type=int, default=5)
    parser.add_argument('--shots-support-test', type=int, default=5)
    parser.add_argument('--shots-query-test', type=int, default=15)
    parser.add_argument('--inner-lr-test', type=float, default=0.001)
    parser.add_argument('--adaptation-steps-test', type=int, default=10)

    args = parser.parse_args()
    main(args)
