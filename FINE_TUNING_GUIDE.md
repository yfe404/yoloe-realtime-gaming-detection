# YOLO-E Fine-Tuning Guide for Gaming Applications

This guide covers fine-tuning YOLO-E models specifically for gaming object detection to improve accuracy and performance.

## 🎯 Why Fine-Tune for Gaming?

Pre-trained YOLO-E models are trained on general datasets (COCO, Open Images) which don't capture:
- **Game-specific art styles** (cartoon, realistic, pixel art, cel-shaded)
- **Unique visual elements** (UI overlays, special effects, lighting)
- **Game-specific objects** (fantasy creatures, sci-fi weapons, specific character designs)
- **Environmental contexts** (indoor maps, outdoor terrains, specific game lighting)

## 📊 Current Performance Issues

- **Low confidence scores** (0.1-0.3 instead of 0.5-0.8+)
- **Poor detection accuracy** on game-specific objects
- **False positives** on UI elements, effects, or similar-looking objects
- **Missed detections** of important game entities

## 🛠️ Fine-Tuning Approaches

### 1. Prompt Engineering (Quickest Start)

**Before Fine-tuning**: Optimize prompts for your specific game

```python
# Generic prompts (poor performance)
PROMPTS = ["enemy soldier", "hostile character"]

# Game-specific prompts (much better)
PROMPTS = [
    "Counter-Strike terrorist with AK-47",
    "enemy player in military fatigues holding rifle",
    "hostile combatant in tactical gear aiming weapon",
    "opponent wearing helmet and body armor"
]
```

**Prompt Optimization Process:**
1. **Analyze your game**: Take 50-100 screenshots of typical enemies
2. **Describe visually**: What do enemies actually look like?
3. **Add context**: Include art style, lighting, perspective
4. **Test iteratively**: Monitor confidence scores and accuracy

### 2. Few-Shot Learning with Custom Classes

**Step 1: Create Custom Dataset**
```bash
# Collect game screenshots
mkdir dataset/
mkdir dataset/images/
mkdir dataset/labels/

# Recommended: 200-500 images per enemy type
# Minimum: 50-100 images per class
```

**Step 2: Annotation Tools**
- **Roboflow**: Web-based, YOLO format export
- **LabelImg**: Desktop tool for bounding boxes
- **CVAT**: Advanced annotation for complex scenes

**Step 3: Dataset Structure**
```
dataset/
├── images/
│   ├── train/          # 70-80% of images
│   ├── val/            # 15-20% of images
│   └── test/           # 5-10% of images
├── labels/
│   ├── train/          # Corresponding YOLO format labels
│   ├── val/
│   └── test/
└── data.yaml          # Dataset configuration
```

**data.yaml example:**
```yaml
# Dataset configuration
path: ./dataset
train: images/train
val: images/val
test: images/test

# Classes (customize for your game)
names:
  0: enemy_soldier
  1: enemy_sniper
  2: friendly_player
  3: neutral_npc
  4: enemy_vehicle

# Number of classes
nc: 5
```

### 3. Transfer Learning Fine-Tuning

**Step 1: Environment Setup**
```bash
# Install requirements
pip install ultralytics wandb  # wandb optional for logging

# GPU requirements
# Minimum: 8GB VRAM (RTX 3070+)
# Recommended: 12GB+ VRAM (RTX 4070+)
```

**Step 2: Training Script**
```python
# fine_tune_yoloe.py
from ultralytics import YOLOE
import torch

def fine_tune_gaming_model():
    # Load pre-trained model
    model = YOLOE('yoloe11l.pt')  # or yoloe11m.pt for faster training
    
    # Training configuration
    results = model.train(
        data='dataset/data.yaml',      # Your custom dataset
        epochs=100,                    # Start with 50-100 epochs
        imgsz=640,                     # Match your inference size
        batch=16,                      # Adjust based on GPU memory
        device=0,                      # GPU device
        
        # Learning rate schedule
        lr0=0.01,                      # Initial learning rate
        lrf=0.1,                       # Final learning rate factor
        warmup_epochs=3,
        
        # Augmentation (important for games)
        hsv_h=0.015,                   # Hue augmentation
        hsv_s=0.7,                     # Saturation augmentation  
        hsv_v=0.4,                     # Value augmentation
        degrees=10,                    # Rotation degrees
        translate=0.1,                 # Translation
        scale=0.5,                     # Scale variation
        fliplr=0.5,                    # Horizontal flip
        
        # Optimization
        optimizer='AdamW',             # Often better than SGD
        weight_decay=0.0005,
        
        # Validation
        val=True,
        save_period=10,                # Save every 10 epochs
        
        # Experiment tracking (optional)
        project='gaming-yoloe',
        name='my-game-v1',
        
        # Early stopping
        patience=30,                   # Stop if no improvement for 30 epochs
        
        # Mixed precision training (faster)
        amp=True,
        
        # Resume training if interrupted
        resume=True
    )
    
    return model, results

if __name__ == "__main__":
    model, results = fine_tune_gaming_model()
    
    # Export optimized model
    model.export(format='onnx')  # For deployment
    print(f"Training completed. Best model saved as: {results.save_dir}/weights/best.pt")
```

**Step 3: Training Monitoring**
```python
# Monitor training progress
def plot_training_results(results_path):
    import matplotlib.pyplot as plt
    
    # Load training results
    results = torch.load(f"{results_path}/results.pt")
    
    # Plot key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0,0].plot(results['train/box_loss'], label='Train Box Loss')
    axes[0,0].plot(results['val/box_loss'], label='Val Box Loss')
    axes[0,0].set_title('Box Loss')
    axes[0,0].legend()
    
    # mAP curves  
    axes[0,1].plot(results['metrics/mAP50'], label='mAP@0.5')
    axes[0,1].plot(results['metrics/mAP50-95'], label='mAP@0.5:0.95')
    axes[0,1].set_title('Mean Average Precision')
    axes[0,1].legend()
    
    # Precision/Recall
    axes[1,0].plot(results['metrics/precision'], label='Precision')
    axes[1,0].plot(results['metrics/recall'], label='Recall')
    axes[1,0].set_title('Precision & Recall')
    axes[1,0].legend()
    
    # Learning rate
    axes[1,1].plot(results['lr/pg0'], label='Learning Rate')
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/training_analysis.png")
    plt.show()
```

### 4. Advanced Fine-Tuning Techniques

**A. Class-Weighted Loss** (for imbalanced data)
```python
# If some enemy types are rare
class_weights = {
    0: 1.0,     # common_enemy
    1: 2.0,     # rare_boss  
    2: 0.5,     # frequent_npc
}
```

**B. Multi-Scale Training**
```python
# Train on multiple resolutions for robustness
imgsz=[416, 512, 640, 768]  # Random selection during training
```

**C. Knowledge Distillation** (for deployment optimization)
```python
# Train large model, then distill to smaller one
teacher_model = YOLOE('yoloe11x.pt')  # Large, accurate model
student_model = YOLOE('yoloe11n.pt')  # Small, fast model

# Implement distillation training loop
```

**D. Pseudo-Labeling**
```python
# Use partially trained model to label more data
def generate_pseudo_labels(model, unlabeled_images, confidence_threshold=0.7):
    results = model.predict(unlabeled_images, conf=confidence_threshold)
    # Convert high-confidence predictions to training labels
    return pseudo_labels
```

## 📈 Expected Improvements After Fine-Tuning

### Detection Quality
- **Confidence scores**: 0.1-0.3 → 0.5-0.8+
- **mAP@0.5**: +20-50% improvement
- **False positives**: 50-80% reduction
- **Missed detections**: 30-60% reduction

### Performance Impact
- **Inference speed**: Similar or slightly faster (optimized weights)
- **Model size**: Same (transfer learning preserves architecture)
- **Memory usage**: Unchanged

## 🚀 Deployment Integration

**Step 1: Replace Model**
```python
# In remote_server_yoloe_ws.py
MODEL_PATH = "path/to/your/fine_tuned_model.pt"

# Your custom classes (no need for set_classes anymore)
# Model already knows your game's objects
```

**Step 2: Update Confidence Threshold**
```python
# Much higher confidence threshold possible
CONF_TH = 0.35  # Instead of 0.10
```

**Step 3: Validation Testing**
```python
# Test on held-out game footage
def validate_model(model_path, test_images):
    model = YOLOE(model_path)
    results = model.val(data='dataset/data.yaml')
    
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map:.3f}")
    
    return results
```

## 🎮 Game-Specific Considerations

### First-Person Shooters (FPS)
- **Focus on**: Weapon detection, player poses, team identification
- **Challenges**: Fast movement, muzzle flash, smoke effects
- **Dataset tips**: Include various lighting, maps, player skins

### Real-Time Strategy (RTS)
- **Focus on**: Unit types, buildings, resource indicators
- **Challenges**: Small objects, top-down view, crowded scenes
- **Dataset tips**: Different zoom levels, unit formations

### MOBA Games
- **Focus on**: Champion identification, spell effects, minions
- **Challenges**: Visual effects, ability animations, team colors
- **Dataset tips**: Different champions, ability states, team skins

### Battle Royale
- **Focus on**: Player detection, loot identification, vehicle types
- **Challenges**: Long-range detection, varied environments, player equipment
- **Dataset tips**: Multiple maps, different player loadouts, range variations

## 🔄 Iterative Improvement Process

1. **Baseline**: Start with prompt optimization
2. **Collect data**: 200-500 images per class minimum
3. **First fine-tune**: Basic transfer learning (50-100 epochs)
4. **Evaluate**: Test on real gameplay
5. **Expand dataset**: Add failure cases, edge cases
6. **Re-train**: Advanced techniques, more epochs
7. **Deploy**: Integration testing
8. **Monitor**: Continuous performance tracking
9. **Update**: Regular model updates with new game content

## 📋 Fine-Tuning Checklist

- [ ] **Data Collection**: 200+ images per enemy type
- [ ] **Annotation**: High-quality bounding boxes
- [ ] **Data Split**: 70% train, 20% val, 10% test
- [ ] **Augmentation**: Game-appropriate transforms
- [ ] **Training**: Monitor loss curves, mAP improvement
- [ ] **Validation**: Test on unseen game footage
- [ ] **Integration**: Update deployment config
- [ ] **A/B Testing**: Compare with baseline model
- [ ] **Documentation**: Training parameters, performance metrics

## 🛠️ Troubleshooting Common Issues

### Low Training mAP
- **Increase epochs**: Try 200-300 for difficult games
- **Reduce learning rate**: Start with lr0=0.001
- **Improve annotations**: Double-check label quality
- **Add more data**: Especially hard examples

### Overfitting
- **Stronger augmentation**: Increase hsv, rotation, scale
- **Reduce model size**: Use yoloe11m instead of yoloe11l
- **Early stopping**: Reduce patience parameter
- **Dropout**: Add regularization

### Slow Convergence
- **Warmup**: Increase warmup_epochs to 5-10
- **Optimizer**: Try different optimizers (Adam, AdamW, SGD)
- **Batch size**: Increase if GPU memory allows
- **Learning rate schedule**: Use cosine annealing

Remember: Fine-tuning is iterative! Start simple, measure improvement, then add complexity as needed.
