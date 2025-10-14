#!/bin/bash

# PIDSE Data Download Script
# Downloads and prepares datasets for PIDSE training

echo "🎯 PIDSE Data Download Script"
echo "================================"

# Create data directories
mkdir -p data/euroc
mkdir -p data/kitti
mkdir -p data/tum

echo "📁 Created data directories"

# Function to download EuRoC dataset
download_euroc() {
    echo "📥 Downloading EuRoC MAV Dataset..."
    
    # EuRoC dataset URLs
    EUROC_BASE="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
    
    # Download selected sequences
    sequences=("MH_01_easy" "MH_02_easy" "MH_03_medium" "V1_01_easy" "V1_02_medium")
    
    for seq in "${sequences[@]}"; do
        echo "  Downloading $seq..."
        
        if [[ $seq == MH* ]]; then
            folder="machine_hall"
        else
            folder="vicon_room1"
        fi
        
        # Download bag file
        wget -P data/euroc/ "$EUROC_BASE/$folder/$seq/$seq.bag" || echo "    ⚠ Failed to download $seq"
    done
    
    echo "✅ EuRoC download completed"
}

# Function to download KITTI dataset
download_kitti() {
    echo "📥 Downloading KITTI Odometry Dataset..."
    
    cd data/kitti
    
    # Download poses
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
    unzip -q data_odometry_poses.zip
    rm data_odometry_poses.zip
    
    # Download velodyne data (selected sequences only)
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
    unzip -q data_odometry_velodyne.zip
    rm data_odometry_velodyne.zip
    
    cd ../..
    echo "✅ KITTI download completed"
}

# Function to download TUM RGB-D dataset
download_tum() {
    echo "📥 Downloading TUM RGB-D Dataset (selected sequences)..."
    
    cd data/tum
    
    # Download selected sequences
    sequences=(
        "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
        "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz"
        "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
    )
    
    for url in "${sequences[@]}"; do
        filename=$(basename "$url")
        echo "  Downloading $filename..."
        wget "$url" || echo "    ⚠ Failed to download $filename"
        tar -xzf "$filename" && rm "$filename"
    done
    
    cd ../..
    echo "✅ TUM download completed"
}

# Main menu
echo ""
echo "Select datasets to download:"
echo "1) EuRoC MAV Dataset (Recommended for drones) - ~2GB"
echo "2) KITTI Odometry Dataset (For ground vehicles) - ~15GB"
echo "3) TUM RGB-D Dataset (For indoor robots) - ~1GB"
echo "4) All datasets - ~18GB"
echo "5) Skip download (use synthetic data only)"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        download_euroc
        ;;
    2)
        download_kitti
        ;;
    3)
        download_tum
        ;;
    4)
        download_euroc
        download_kitti
        download_tum
        ;;
    5)
        echo "⏭ Skipping download - will use synthetic data"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Data setup completed!"
echo "📊 Data summary:"
ls -la data/*/

echo ""
echo "🚀 Next steps:"
echo "1. Install ROS tools to convert EuRoC .bag files (if downloaded)"
echo "2. Run: python examples/quick_start.py"
echo "3. Or train with real data: python experiments/train_pidse.py --config experiments/configs/real_data_config.yaml"