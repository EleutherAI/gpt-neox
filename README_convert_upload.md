# GPT-NeoX to HuggingFace Conversion & Upload Script

This script automates the process of converting GPT-NeoX checkpoints to HuggingFace format and uploading them to the HuggingFace Hub.

## Features

- ✅ **Automated conversion** from NeoX to HuggingFace format
- ✅ **Direct upload** to HuggingFace Hub
- ✅ **Model loading test** to verify successful upload
- ✅ **Error handling** and validation at each step
- ✅ **Progress tracking** with detailed logging
- ✅ **Flexible options** for different use cases

## Prerequisites

1. **HuggingFace CLI login**:
   ```bash
   huggingface-cli login
   ```

2. **Required packages**:
   ```bash
   pip install transformers torch huggingface-hub
   ```

## Usage

### Basic Usage
```bash
python convert_and_upload.py <experiment_name> <checkpoint_number> <hf_repo_name>
```

### Examples

**Convert and upload the deep fry 20x model:**
```bash
python convert_and_upload.py annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653 Unlearning/pythia1.5_modernbert_filtered_5percent_wmdp_deep_fry_20x_upsampled
```

**Convert only (no upload):**
```bash
python convert_and_upload.py annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653 Unlearning/test-model --no-upload
```

**Convert and upload but skip testing:**
```bash
python convert_and_upload.py annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653 Unlearning/test-model --no-test
```

### Options

- `--no-upload`: Skip uploading to HuggingFace Hub (convert only)
- `--no-test`: Skip testing model loading after upload
- `--force-convert`: Force reconversion even if output already exists

## What the Script Does

### 1. **Validation Phase**
- ✅ Checks that the NeoX checkpoint exists
- ✅ Verifies HuggingFace CLI login status
- ✅ Validates required files and permissions

### 2. **Conversion Phase**
- 🔄 Converts NeoX checkpoint to HuggingFace format
- 📁 Saves to `/checkpoints/hf_converted/{experiment_name}/global_step{checkpoint_number}/`
- ✅ Verifies essential files (config.json, tokenizer.json, model shards)

### 3. **Upload Phase** (if not skipped)
- 📤 Uploads all model files to HuggingFace Hub
- 🔗 Provides direct link to uploaded model
- ⏱️ Handles large file uploads with extended timeout

### 4. **Testing Phase** (if not skipped)
- 🧪 Tests model loading with transformers
- ✅ Verifies Flash Attention 2 compatibility
- 📊 Reports model statistics (parameters, dtype, device)

## File Structure

After successful conversion, you'll have:

```
/checkpoints/hf_converted/{experiment_name}/global_step{checkpoint_number}/
├── config.json                    # Model configuration
├── generation_config.json         # Generation settings
├── model-00001-of-00006.safetensors  # Model weights (sharded)
├── model-00002-of-00006.safetensors
├── ...
├── model.safetensors.index.json   # Shard index
├── special_tokens_map.json        # Special tokens
├── tokenizer.json                 # Tokenizer
└── tokenizer_config.json          # Tokenizer config
```

## Error Handling

The script includes comprehensive error handling:

- **Checkpoint validation**: Ensures source checkpoint exists
- **Conversion verification**: Checks output files after conversion
- **Upload validation**: Verifies HuggingFace login and permissions
- **Loading test**: Confirms model can be loaded successfully
- **Timeout protection**: Prevents hanging on large uploads

## Troubleshooting

### Common Issues

1. **"Not logged in to HuggingFace"**
   ```bash
   huggingface-cli login
   ```

2. **"Checkpoint path does not exist"**
   - Check the experiment name spelling
   - Verify the checkpoint is in `/checkpoints/`

3. **"Conversion failed"**
   - Check disk space
   - Verify checkpoint integrity
   - Check config file path

4. **Upload timeout**
   - Script automatically uses 2-hour timeout
   - For very large models, consider uploading manually

### Getting Help

Run with `--help` for detailed usage information:
```bash
python convert_and_upload.py --help
```

## Integration with Existing Workflows

This script can be easily integrated into training pipelines:

```bash
# After training completes
python convert_and_upload.py $EXPERIMENT_NAME $FINAL_STEP $HF_REPO_NAME

# For evaluation workflows
python convert_and_upload.py $EXPERIMENT_NAME $STEP $HF_REPO_NAME --no-test
```