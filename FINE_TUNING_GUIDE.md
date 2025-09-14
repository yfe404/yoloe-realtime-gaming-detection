# YOLO-E Fine-Tuning Guide for Gaming Applications

This guide covers fine-tuning YOLO-E models specifically for gaming object detection to improve accuracy and performance.

## 🎯 Why Fine-Tune YOLO-E for Gaming?

Pre-trained YOLO-E models are trained on general datasets (Objects365, LVIS, GoldG, Flickr30k) which don't capture:
- **Game-specific art styles** (cartoon, realistic, pixel art, cel-shaded)  
- **Unique visual elements** (UI overlays, special effects, lighting)
- **Game-specific objects** (fantasy creatures, sci-fi weapons, specific character designs)
- **Environmental contexts** (indoor maps, outdoor terrains, specific game lighting)

## 📊 Current Performance Issues

- **Low confidence scores** (0.1-0.3 instead of 0.5-0.8+)
- **Poor detection accuracy** on game-specific objects
- **False positives** on UI elements, effects, or similar-looking objects
- **Missed detections** of important game entities

## 🛠️ YOLO-E Specific Fine-Tuning Approaches

### 1. Model Selection and Architecture

**Available YOLO-E Models:**
```python
# Text/Visual Prompt Models (recommended for gaming)
"yoloe-11s-seg.pt"    # 70MB, fastest
"yoloe-11m-seg.pt"    # Medium speed/accuracy
"yoloe-11l-seg.pt"    # 67MB, best accuracy

# Prompt-Free Models (4,585 built-in classes)
"yoloe-11s-seg-pf.pt" # Good for general object detection
"yoloe-11m-seg-pf.pt"
"yoloe-11l-seg-pf.pt"
```

**For Gaming Applications:**
- Use **text/visual prompt models** for custom enemy types
- Consider **prompt-free models** for general object detection
- **Detection-only models**: Convert segmentation models if you only need bounding boxes

### 2. Quick Start: Text Prompt Optimization (Zero Training)

**Before Fine-tuning**: Optimize prompts for immediate improvement

```python
from ultralytics import YOLOE

# Load model
model = YOLOE("yoloe-11l-seg.pt")

# Game-specific prompts (much better than generic)
game_prompts = [
    "enemy soldier in tactical gear with rifle",
    "hostile player in combat armor",
    "armed opponent with assault weapon",
    "enemy combatant in military uniform"
]

# Set classes (this replaces generic LVIS classes)
model.set_classes(game_prompts, model.get_text_pe(game_prompts))

# Run inference
results = model.predict("game_screenshot.jpg", conf=0.25)
```

**Expected improvement**: 50-200% better detection without any training

### 3. Fine-Tuning with Custom Gaming Dataset

**Step 1: Trainer Selection**
YOLO-E uses specialized trainers:

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer as Trainer

# For instance segmentation fine-tuning
model = YOLOE("yoloe-11s-seg.pt")
results = model.train(
    data="gaming_dataset.yaml",
    epochs=80,
    trainer=Trainer,  # Essential for YOLO-E
    # ... other parameters
)
```

**Step 2: YOLO-E Specific Training Parameters**
```python
results = model.train(
    data="gaming_dataset.yaml",
    epochs=80,
    close_mosaic=10,          # YOLO-E specific: disable mosaic near end
    batch=16,
    optimizer="AdamW",        # Recommended for YOLO-E
    lr0=1e-3,                 # Learning rate
    warmup_bias_lr=0.0,       # YOLO-E specific
    weight_decay=0.025,       # YOLO-E specific
    momentum=0.9,
    workers=4,
    device="0",
    trainer=Trainer,
)
```

### 4. Linear Probing (Limited Data)

When you have limited gaming data, use linear probing to fine-tune only the classification head:

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer as Trainer

model = YOLOE("yoloe-11s-seg.pt")

# Freeze everything except classification head
head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]

# Freeze specific head components except classification
for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

# Freeze detection branch components
freeze.extend([
    f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1",
    f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1", 
    f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1",
])

results = model.train(
    data="gaming_dataset.yaml",
    epochs=50,  # Fewer epochs for linear probing
    trainer=Trainer,
    freeze=freeze,
    # ... other parameters
)
```

### 5. Detection-Only Model Training

Convert segmentation model to detection for better performance:

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer as Trainer

# Create detection model from YAML, load segmentation weights
model = YOLOE("yoloe-11s.yaml").load("yoloe-11s-seg.pt")

# Train as detection model (faster than segmentation)
results = model.train(
    data="gaming_dataset.yaml",
    epochs=80,
    trainer=Trainer,  # Use YOLOEPETrainer for detection
    # ... other parameters
)
```

## 📈 Gaming-Specific Optimizations

### 1. Dataset Preparation for Gaming
```yaml
# gaming_dataset.yaml
path: ./gaming_dataset
train: images/train
val: images/val
test: images/test

# Gaming-specific classes
names:
  0: enemy_soldier
  1: friendly_player
  2: enemy_vehicle
  3: hostile_npc
  4: neutral_object

nc: 5
```

### 2. Gaming Data Augmentation
```python
results = model.train(
    # Gaming-specific augmentations
    hsv_h=0.015,      # Hue variation for different lighting
    hsv_s=0.7,        # Saturation for different graphics settings
    hsv_v=0.4,        # Brightness for day/night cycles
    degrees=5,        # Less rotation (games have consistent perspective)
    translate=0.1,    # Small translation
    scale=0.3,        # Scale variation for zoom levels
    fliplr=0.5,       # Horizontal flip OK for most games
    flipud=0.0,       # Usually don't flip vertically in games
)
```

### 3. Visual Prompt Training
For few-shot learning with visual examples:

```python
from ultralytics.models.yolo.yoloe import YOLOESegVPTrainer

# Visual prompt training (advanced)
model = YOLOE("yoloe-11l-seg.pt")

# Freeze everything except SAVPE module
head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if "savpe" not in name:
        freeze.append(f"{head_index}.{name}")

model.train(
    data=training_data,
    trainer=YOLOESegVPTrainer,  # Visual prompt trainer
    freeze=freeze,
    epochs=20,  # Usually needs fewer epochs
    lr0=16e-3,  # Higher learning rate for visual prompts
)
```

## 🎮 Game-Specific Considerations

### FPS Games (Counter-Strike, Call of Duty)
```python
fps_prompts = [
    "terrorist with AK-47 rifle",
    "counter-terrorist with M4A4",
    "enemy player in tactical vest",
    "hostile combatant with kevlar armor",
    "sniper with scoped rifle"
]
```

### MOBA Games (League of Legends, Dota 2)
```python
moba_prompts = [
    "enemy champion in team fight",
    "hostile minion in lane",
    "enemy tower structure", 
    "opponent jungler with abilities",
    "enemy support character"
]
```

### Battle Royale (Fortnite, PUBG)
```python
br_prompts = [
    "enemy player with assault rifle",
    "hostile opponent in building",
    "enemy squad member",
    "opponent with sniper rifle",
    "enemy player in vehicle"
]
```

## 🔄 Deployment Integration

**Update Server Configuration:**
```python
# In remote_server_yoloe_ws.py
MODEL_PATH = "path/to/your/fine_tuned_yoloe_model.pt"

# Higher confidence threshold with fine-tuned model
CONF_TH = 0.35  # Instead of 0.10

# Custom game classes automatically loaded
# No need for set_classes() with fine-tuned models
```

**Performance Monitoring:**
```python
def validate_gaming_model(model_path, test_data):
    model = YOLOE(model_path)
    
    # Validate on gaming test set
    results = model.val(data="gaming_dataset.yaml")
    
    print(f"Gaming mAP@0.5: {results.box.map50:.3f}")
    print(f"Gaming mAP@0.5:0.95: {results.box.map:.3f}")
    
    return results
```

## 📋 Expected Improvements After Fine-Tuning

### Detection Quality
- **Confidence scores**: 0.1-0.3 → 0.5-0.8+
- **mAP@0.5**: +30-60% improvement on gaming scenes
- **False positives**: 60-80% reduction
- **Missed detections**: 40-70% reduction

### Integration Benefits
- **Higher confidence threshold**: Cleaner detections
- **Game-specific understanding**: Better context awareness
- **Reduced false alarms**: Less noise from UI elements

## 🚨 Common YOLO-E Fine-Tuning Pitfalls

### 1. Wrong Trainer Class
```python
# ❌ Wrong - using generic YOLO trainer
from ultralytics import YOLO
model = YOLO("yoloe-11s-seg.pt")  # This won't work properly

# ✅ Correct - using YOLO-E trainer
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
model = YOLOE("yoloe-11s-seg.pt")
model.train(trainer=YOLOEPESegTrainer)
```

### 2. Missing YOLO-E Specific Parameters
```python
# ✅ Include YOLO-E specific parameters
results = model.train(
    close_mosaic=10,        # Essential for YOLO-E
    warmup_bias_lr=0.0,     # YOLO-E specific
    weight_decay=0.025,     # Different from standard YOLO
    optimizer="AdamW",      # Recommended for YOLO-E
)
```

### 3. Prompt Management After Fine-Tuning
```python
# After fine-tuning, your model knows custom classes
# No need to set_classes() for fine-tuned classes

# ❌ Don't do this with fine-tuned models
model.set_classes(["enemy"], model.get_text_pe(["enemy"]))

# ✅ Fine-tuned model automatically recognizes trained classes
results = model.predict("game_screenshot.jpg")
```

Remember: YOLO-E fine-tuning is more complex than standard YOLO but offers powerful open-vocabulary capabilities. Start with prompt optimization, then move to fine-tuning for maximum gaming performance.
