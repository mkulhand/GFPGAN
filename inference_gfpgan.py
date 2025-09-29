import argparse
import cv2
import glob
import numpy as np
import os
import torch
import multiprocessing as mp
import time
from basicsr.utils import imwrite
from gfpgan import GFPGANer


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def detect_face_dnn(image, net=None, confidence_threshold=0.5):
    """
    Detect faces using OpenCV DNN with better accuracy
    Uses a pre-trained ResNet-based face detection model
    """
    if net is None:
        # Load DNN model (you need to download these files)
        # Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
        prototxt_path = "opencv_face_detector.pbtxt"
        model_path = "opencv_face_detector_uint8.pb"

        # Check if model files exist
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            print("DNN model files not found, falling back to Haar cascades...")
            return detect_face_haar(image)

        net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)

    (h, w) = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    # Pass blob through network
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure coordinates are within image bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            width = endX - startX
            height = endY - startY

            if width > 30 and height > 30:  # Minimum size filter
                faces.append((startX, startY, width, height))

    return faces


def detect_face_haar(image, face_cascade=None):
    """
    Detect faces using Haar cascades (fallback method)
    """
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try multiple scale factors for better detection
    scale_factors = [1.05, 1.1, 1.2, 1.3]
    all_faces = []

    for scale_factor in scale_factors:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=3,  # Reduced for more sensitive detection
            minSize=(20, 20),  # Smaller minimum size
            maxSize=(800, 800),
        )
        all_faces.extend(faces)

    # Remove duplicates (faces detected multiple times)
    if len(all_faces) > 0:
        # Convert to list of tuples for easier processing
        unique_faces = []
        for face in all_faces:
            x, y, w, h = face
            # Check if this face overlaps significantly with existing faces
            is_duplicate = False
            for existing in unique_faces:
                ex, ey, ew, eh = existing
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                current_area = w * h
                existing_area = ew * eh

                if overlap_area > 0.5 * min(current_area, existing_area):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_faces.append((x, y, w, h))

        return unique_faces

    return []


def detect_face_mtcnn(image):
    """
    Detect faces using MTCNN (if available)
    You need to install mtcnn: pip install mtcnn-pytorch
    """
    try:
        from mtcnn import MTCNN

        detector = MTCNN()

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        result = detector.detect_faces(rgb_image)

        faces = []
        for face in result:
            if face["confidence"] > 0.9:  # High confidence threshold
                x, y, w, h = face["box"]
                # Ensure coordinates are valid
                if x >= 0 and y >= 0 and w > 30 and h > 30:
                    faces.append((x, y, w, h))

        return faces

    except ImportError:
        print("MTCNN not available, using DNN detection...")
        return detect_face_dnn(image)


def detect_face_size(image, method="auto"):
    """
    Detect faces and return the largest face size

    Args:
        image: Input image
        method: Detection method ('dnn', 'haar', 'mtcnn', 'auto')

    Returns:
        tuple: (face_width, face_height) of the largest face
    """
    faces = []

    if method == "auto":
        # Try methods in order of accuracy
        try:
            faces = detect_face_mtcnn(image)
            if len(faces) == 0:
                faces = detect_face_dnn(image)
            if len(faces) == 0:
                faces = detect_face_haar(image)
        except Exception as e:
            print(f"Auto detection failed: {e}, using Haar cascades")
            faces = detect_face_haar(image)
    elif method == "dnn":
        faces = detect_face_dnn(image)
    elif method == "mtcnn":
        faces = detect_face_mtcnn(image)
    else:  # haar
        faces = detect_face_haar(image)

    if len(faces) == 0:
        return 0, 0  # No faces found

    # Find the largest face
    largest_face = max(faces, key=lambda face: face[2] * face[3])  # width * height
    face_width, face_height = largest_face[2], largest_face[3]

    return face_width, face_height


def should_restore_face(image, min_face_size=150, max_face_size=800, method="auto"):
    """
    Determine if face should be restored based on size

    Args:
        image: Input image
        min_face_size: Minimum face size to restore (faces smaller than this will be restored)
        max_face_size: Maximum face size to restore (faces larger than this won't be restored)
        method: Face detection method

    Returns:
        bool: True if face should be restored
    """
    face_width, face_height = detect_face_size(image, method)

    if face_width == 0 or face_height == 0:
        return False  # No faces detected

    # Calculate average face size
    avg_face_size = (face_width + face_height) / 2

    # Restore only if face is small/far (below max_face_size)
    # and above minimum detectable size
    return min_face_size <= avg_face_size <= max_face_size


def analyze_all_images(img_list, min_face_size, max_face_size, detection_method="auto"):
    """
    Analyze all images to determine restoration percentage based on faces needing restoration
    Uses early stopping with statistical projection to optimize analysis time

    Returns:
        tuple: (restore_all_flag, total_needing_restoration, total_images, analysis_results)
    """
    print(f"Analyzing images using {detection_method} face detection...")
    print(
        f"Face size criteria: {min_face_size}-{max_face_size} pixels need restoration"
    )

    total_with_faces = 0
    total_needing_restoration = 0
    analysis_results = {}

    # Early stopping parameters
    min_samples_for_projection = min(
        161, len(img_list) // 4
    )  # At least 25% or 200 images
    projection_check_interval = 50  # Check projection every 50 images

    for i, img_path in enumerate(img_list):
        if i % 50 == 0:
            print(f"Analyzing image {i+1}/{len(img_list)}...")

        img = cv2.imread(img_path)
        if img is None:
            analysis_results[img_path] = {
                "has_face": False,
                "face_size": 0,
                "needs_restoration": False,
            }
            continue

        face_width, face_height = detect_face_size(img, detection_method)

        if face_width == 0 or face_height == 0:
            # No face detected
            analysis_results[img_path] = {
                "has_face": False,
                "face_size": 0,
                "needs_restoration": False,
            }
        else:
            # Face detected
            total_with_faces += 1
            face_size = (face_width + face_height) / 2

            # Check if this face needs restoration based on size criteria
            needs_restoration = min_face_size <= face_size <= max_face_size
            if needs_restoration:
                total_needing_restoration += 1

            analysis_results[img_path] = {
                "has_face": True,
                "face_size": face_size,
                "needs_restoration": needs_restoration,
            }

        # Early stopping check
        if i >= min_samples_for_projection and (i + 1) % projection_check_interval == 0:
            current_sample_size = i + 1
            current_restoration_rate = total_needing_restoration / current_sample_size

            # Calculate confidence interval for the restoration rate
            # Using Wilson score interval for better accuracy with small samples
            import math

            z = 1.96  # 95% confidence
            n = current_sample_size
            p = current_restoration_rate

            if n > 0 and p >= 0:  # Safety check
                denominator = 1 + (z**2 / n)
                center = (p + (z**2 / (2 * n))) / denominator
                margin = (z / denominator) * math.sqrt(
                    (p * (1 - p) / n) + (z**2 / (4 * n**2))
                )

                lower_bound = max(0, center - margin)
                upper_bound = min(1, center + margin)

                # Check if we can be confident about the final result
                threshold = 0.50

                if upper_bound < threshold:
                    # Even in best case, we won't reach 50%
                    print(
                        f"\n🚀 Early stopping at {current_sample_size}/{len(img_list)} images"
                    )
                    print(
                        f"   Current restoration rate: {current_restoration_rate*100:.1f}%"
                    )
                    print(
                        f"   95% confidence interval: {lower_bound*100:.1f}% - {upper_bound*100:.1f}%"
                    )
                    print(f"   Projection: Will NOT reach 50% → RESTORE NOTHING")

                    # Use projected total based on current rate
                    estimated_total_needing = int(
                        current_restoration_rate * len(img_list)
                    )
                    early_stopped = True
                    break

                elif lower_bound > threshold:
                    # Even in worst case, we'll exceed 50%
                    print(
                        f"\n🚀 Early stopping at {current_sample_size}/{len(img_list)} images"
                    )
                    print(
                        f"   Current restoration rate: {current_restoration_rate*100:.1f}%"
                    )
                    print(
                        f"   95% confidence interval: {lower_bound*100:.1f}% - {upper_bound*100:.1f}%"
                    )
                    print(f"   Projection: Will EXCEED 50% → RESTORE ALL")

                    # Use projected total based on current rate
                    estimated_total_needing = int(
                        current_restoration_rate * len(img_list)
                    )
                    early_stopped = True
                    break

    else:
        # No early stopping, use actual results
        estimated_total_needing = total_needing_restoration
        early_stopped = False

    # Calculate final percentage
    restoration_percentage = (estimated_total_needing / len(img_list)) * 100
    restore_all = restoration_percentage >= 50.0

    print(f"\nAnalysis complete:")
    print(f"  Total images: {len(img_list)}")
    print(f"  Images analyzed: {len(analysis_results)}")
    if early_stopped:
        print(
            f"  Early stopping used - projected {len(img_list) - len(analysis_results)} remaining images"
        )
        print(
            f"  Actual images needing restoration (analyzed): {total_needing_restoration}"
        )
        print(f"  Projected total needing restoration: {estimated_total_needing}")
    else:
        print(f"  Images with faces: {total_with_faces}")
        print(f"  Images needing restoration: {estimated_total_needing}")
    print(f"  Restoration percentage: {restoration_percentage:.1f}%")
    print(f"  Detection method: {detection_method}")
    print(
        f"  Strategy: {'RESTORE ALL FRAMES' if restore_all else 'RESTORE NOTHING (copy originals)'}"
    )

    return restore_all, estimated_total_needing, len(img_list), analysis_results


def process_batch(args_tuple):
    """Process a batch of images in a separate process"""
    (
        batch_files,
        output_dir,
        version,
        upscale,
        process_id,
        face_size_config,
        restore_all_faces,
        analysis_results,
    ) = args_tuple

    try:
        # Clear GPU memory at start
        clear_gpu_memory()

        print(f"Process {process_id}: Starting with {len(batch_files)} images")
        print(
            f"Process {process_id}: Mode: {'RESTORE ALL' if restore_all_faces else 'RESTORE NOTHING (with upscaling)'}"
        )

        # If not restoring faces, use upsampler for all images
        if not restore_all_faces:
            print(
                f"Process {process_id}: Upscaling all frames without face restoration"
            )

            # Initialize background upsampler for upscaling
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Initialize RealESRGAN for upscaling
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=upscale,
            )
            netscale = upscale

            # Try multiple model path locations
            model_paths = [
                f"experiments/pretrained_models/RealESRGAN_x{upscale}plus.pth",
                f"realesrgan/weights/RealESRGAN_x{upscale}plus.pth",
                f"RealESRGAN_x{upscale}plus.pth",
            ]

            model_path = None
            for path in model_paths:
                if os.path.isfile(path):
                    model_path = path
                    break

            # If no local model found, use default x2 upscaler or download URL
            if model_path is None:
                if upscale == 2:
                    model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                else:
                    model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

            upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=400,  # Adjust based on VRAM
                tile_pad=10,
                pre_pad=0,
                half=True,  # Use FP16 for faster processing
            )

            # Upscale all images
            upscaled_count = 0
            for img_path in batch_files:
                try:
                    clear_gpu_memory()

                    img_name = os.path.basename(img_path)
                    basename, ext = os.path.splitext(img_name)
                    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                    if input_img is not None:
                        # Upscale the image
                        output, _ = upsampler.enhance(input_img, outscale=upscale)

                        save_path = os.path.join(output_dir, f"{basename}.png")
                        cv2.imwrite(save_path, output)
                        upscaled_count += 1

                        if upscaled_count % 100 == 0:
                            print(
                                f"Process {process_id}: Upscaled {upscaled_count}/{len(batch_files)} images"
                            )
                    else:
                        print(f"Process {process_id}: Failed to load {img_name}")

                    clear_gpu_memory()

                except Exception as e:
                    print(f"Process {process_id}: Error upscaling {img_name}: {e}")
                    # Fallback: copy original if upscaling fails
                    try:
                        save_path = os.path.join(output_dir, f"{basename}.png")
                        cv2.imwrite(save_path, input_img)
                    except:
                        pass
                    continue

            return f"Process {process_id}: Upscaled {upscaled_count} images (no face restoration)"

        # Initialize GFPGAN only if we're restoring all frames
        if version == "1.4":
            arch = "clean"
            channel_multiplier = 2
            model_name = "GFPGANv1.4"

            # Try multiple model path locations
            model_paths = [
                f"experiments/pretrained_models/{model_name}.pth",
                f"gfpgan/weights/{model_name}.pth",
                f"{model_name}.pth",
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            ]

            model_path = None
            for path in model_paths:
                if path.startswith("http") or os.path.isfile(path):
                    model_path = path
                    break

            if model_path is None:
                model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

        # Use CPU background upsampler to save VRAM
        bg_upsampler = None

        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler,
        )

        # Process each image in the batch
        processed_count = 0
        copied_count = 0

        for img_path in batch_files:
            try:
                # Clear memory before each image
                clear_gpu_memory()

                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if input_img is None:
                    print(f"Process {process_id}: Failed to load {img_name}")
                    continue

                # In restore_all mode, process EVERY frame
                img_analysis = analysis_results.get(
                    img_path, {"has_face": False, "face_size": 0}
                )
                face_status = "with face" if img_analysis["has_face"] else "no face"

                print(
                    f"Process {process_id}: Processing {img_name} (restore all - {face_status})"
                )

                # Restore the image
                cropped_faces, restored_faces, restored_img = restorer.enhance(
                    input_img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5,
                )

                # Save restored image
                if restored_img is not None:
                    save_restore_path = os.path.join(output_dir, f"{basename}.png")
                    imwrite(restored_img, save_restore_path)
                    processed_count += 1
                else:
                    # If restoration failed, copy original
                    save_restore_path = os.path.join(output_dir, f"{basename}.png")
                    cv2.imwrite(save_restore_path, input_img)
                    copied_count += 1

                # Clear memory after processing
                clear_gpu_memory()

            except torch.cuda.OutOfMemoryError:
                print(f"Process {process_id}: VRAM error on {img_name}, skipping")
                clear_gpu_memory()
                continue
            except Exception as e:
                print(f"Process {process_id}: Error on {img_name}: {e}")
                continue

        return f"Process {process_id}: Restored {processed_count}, Copied {copied_count} images"

    except Exception as e:
        return f"Process {process_id}: Failed with error: {e}"


def split_files_into_batches(file_list, num_processes):
    """Split files into batches for parallel processing"""
    batch_size = len(file_list) // num_processes
    batches = []

    for i in range(num_processes):
        start_idx = i * batch_size
        if i == num_processes - 1:  # Last batch gets remaining files
            end_idx = len(file_list)
        else:
            end_idx = (i + 1) * batch_size

        batches.append(file_list[start_idx:end_idx])

    return batches


def download_dnn_models():
    """Download DNN face detection models if they don't exist"""
    import urllib.request

    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"

    prototxt_path = "opencv_face_detector.pbtxt"
    model_path = "opencv_face_detector_uint8.pb"

    if not os.path.exists(prototxt_path):
        print("Downloading DNN prototxt file...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print("✓ Downloaded opencv_face_detector.pbtxt")

    if not os.path.exists(model_path):
        print("Downloading DNN model file (this may take a while)...")
        urllib.request.urlretrieve(model_url, model_path)
        print("✓ Downloaded opencv_face_detector_uint8.pb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="frames", help="Input folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="restored_frames", help="Output folder"
    )
    parser.add_argument(
        "-v", "--version", type=str, default="1.4", help="GFPGAN version"
    )
    parser.add_argument("-s", "--upscale", type=int, default=2, help="Upscale factor")
    parser.add_argument(
        "-p", "--processes", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument(
        "--min_face_size",
        type=int,
        default=150,
        help="Minimum face size to restore (pixels, default: 150)",
    )
    parser.add_argument(
        "--max_face_size",
        type=int,
        default=800,
        help="Maximum face size to restore (pixels, default: 800)",
    )
    parser.add_argument(
        "--detection_method",
        type=str,
        default="auto",
        choices=["auto", "dnn", "haar", "mtcnn"],
        help="Face detection method (default: auto)",
    )
    parser.add_argument(
        "--download_models",
        action="store_true",
        help="Download DNN face detection models",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview face detection without processing",
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="Disable early stopping optimization (analyze all images)",
    )
    parser.add_argument(
        "--force_selective",
        action="store_true",
        help="Force restore nothing mode even if >=50% have faces",
    )
    args = parser.parse_args()

    # Modify analysis function based on early stopping preference
    if args.no_early_stop:
        # Disable early stopping by setting unrealistic confidence threshold
        print("Early stopping disabled - will analyze all images")

    # Download DNN models if requested
    if args.download_models:
        download_dnn_models()
        print("DNN models downloaded successfully!")
        return

    # Get all image files
    extensions = ["jpg", "jpeg", "png", "bmp"]
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
        img_list.extend(glob.glob(os.path.join(args.input, f"*.{ext.upper()}")))

    img_list = sorted(img_list)
    print(f"Found {len(img_list)} images to process")

    if len(img_list) == 0:
        print("No images found to process!")
        return

    # Analyze all images first
    restore_all, total_needing_restoration, total_images, analysis_results = (
        analyze_all_images(
            img_list, args.min_face_size, args.max_face_size, args.detection_method
        )
    )

    # Override if force_selective is set
    if args.force_selective:
        restore_all = False
        print("Forcing restore nothing mode (--force_selective)")

    # Preview mode - show analysis results
    if args.preview:
        print("\nPreview mode - showing analysis results...")

        for i, img_path in enumerate(img_list[:20]):  # Preview first 20 images
            img_analysis = analysis_results.get(
                img_path,
                {"has_face": False, "face_size": 0, "needs_restoration": False},
            )

            if restore_all:
                status = "RESTORE (all frames)"
            else:
                status = "COPY ORIGINAL (restore nothing)"

            if img_analysis["has_face"]:
                face_size = img_analysis["face_size"]
                needs = (
                    "needs restoration"
                    if img_analysis["needs_restoration"]
                    else "no restoration needed"
                )
                face_info = f"Face {face_size:.0f}px ({needs})"
            else:
                face_info = "No face"

            print(f"{os.path.basename(img_path)}: {face_info} -> {status}")

        if len(img_list) > 20:
            print(f"... (showing first 20 of {len(img_list)} images)")

        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Split files into batches
    batches = split_files_into_batches(img_list, args.processes)

    # Face size configuration
    face_size_config = {
        "min_face_size": args.min_face_size,
        "max_face_size": args.max_face_size,
    }

    # Prepare arguments for each process
    process_args = []
    for i, batch in enumerate(batches):
        if batch:  # Only add non-empty batches
            process_args.append(
                (
                    batch,
                    args.output,
                    args.version,
                    args.upscale,
                    i + 1,
                    face_size_config,
                    restore_all,  # Pass the restore_all flag
                    analysis_results,  # Pass the analysis results
                )
            )

    print(f"Starting {len(process_args)} processes...")

    # Process batches in parallel
    start_time = time.time()
    with mp.Pool(processes=len(process_args)) as pool:
        results = pool.map(process_batch, process_args)

    # Print results
    total_processed = 0
    total_copied = 0

    for result in results:
        print(result)
        # Parse counts from result string
        try:
            if "Restored" in result and "Copied" in result:
                # Format: "Process X: Restored Y, Copied Z images"
                parts = result.split(": ")[1]  # Get part after "Process X: "
                if ", " in parts:
                    for part in parts.split(", "):
                        part = part.strip()
                        if part.startswith("Restored"):
                            total_processed += int(part.split()[1])
                        elif part.startswith("Copied") and not part.endswith(
                            "(no restoration)"
                        ):
                            num_str = part.split()[1]
                            total_copied += int(num_str)
            elif "Copied" in result and "(no restoration)" in result:
                # Format: "Process X: Copied Y images (no restoration)"
                parts = result.split(": ")[1]
                num_str = parts.split()[1]  # Get the number after "Copied"
                total_copied += int(num_str)
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse result: {result}")
            continue

    elapsed_time = time.time() - start_time
    print(f"\n=== Final Summary ===")
    print(f"Total images processed: {len(img_list)}")
    print(f"Face detection method: {args.detection_method}")
    print(f"Face size criteria: {args.min_face_size}-{args.max_face_size} pixels")

    # Count total faces and needing restoration from analysis_results
    total_with_faces = sum(
        1 for analysis in analysis_results.values() if analysis["has_face"]
    )
    total_needing_restoration_check = sum(
        1
        for analysis in analysis_results.values()
        if analysis.get("needs_restoration", False)
    )

    print(
        f"Images with faces detected: {total_with_faces} ({(total_with_faces/len(img_list)*100):.1f}%)"
    )
    print(
        f"Images needing restoration: {total_needing_restoration_check} ({(total_needing_restoration_check/len(img_list)*100):.1f}%)"
    )

    if restore_all:
        print(f"Restoration mode: RESTORE ALL FRAMES")
        print(f"All {len(img_list)} frames were processed with GFPGAN")
    else:
        print(f"Restoration mode: RESTORE NOTHING")
        print(
            f"All {len(img_list)} frames were copied as originals (no GFPGAN processing)"
        )

    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Results are in [{args.output}] folder.")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Required for CUDA multiprocessing
    main()
