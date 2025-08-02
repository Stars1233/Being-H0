# Camera Intrinsics for Mobile Phone Photos

This guide helps you estimate camera intrinsic parameters when using your own mobile phone photos for BeingVLA inference.

## Quick Estimation Method

### 1. Search your camera's specifications

You may extract photo's EXIF metadata or search on the web to find camera specifications.

### 2. Simple Formula

```bash
# Get focal length (mm) from EXIF
focal_length_mm = exif_data['FocalLength']

# Extract EXIF info or search mobile phone sensor specifications on the web
sensor_width_mm = 5.76  # Example for a phone sensor
sensor_height_mm = 4.29

# Calculate intrinsic parameters
fx = focal_length_mm * image_width / sensor_width_mm
fy = focal_length_mm * image_height / sensor_height_mm
cx = image_width / 2
cy = image_height / 2
```

### 3. Inference with Estimated Intrinsics

If you have estimated the intrinsic parameters (fx fy cx cy), you can use them in the inference command:

```bash
python -m beingvla.inference.vla_internvl_inference \
    --input_intrinsic fx fy cx cy \
    --model_path /path/to/Being-H0-XXX \
    --motion_code_path "/path/to/Being-H0-GRVQ-8K/wrist/+/path/to/Being-H0-GRVQ-8K/finger/" \
    --input_image ./path/to/your/image \
    --task_description "task description" \
    --hand_mode <hand mode (left/right/both)> \
    --num_samples 3 \
    --num_seconds 4 \
    --enable_render true \
    --gpu_device 0 \
    --output_dir ./work_dirs/
```

## Tips for Best Results

1. **Camera Selection**
   - Use the main (wide) camera, not telephoto or ultra-wide
   - Avoid digital zoom or cropping
   - Use default camera app settings

2. **Photo Capture**
   - Take photos at arm's length distance (0.5-1.5 meters)
   - Try to capture scene with human hand, which will help the model get a better initialization. However, it's OK to use photos without hands.

## Understanding Camera Intrinsics (Optional)

Camera intrinsic parameters define how 3D points project onto the 2D image:
- **fx, fy**: Focal length in pixels (horizontal and vertical)
- **cx, cy**: Principal point (optical center) in pixels

For mobile phones, we can estimate these because:
- Most phones have similar field of view (60-80 degrees)
- Principal point is usually near image center
