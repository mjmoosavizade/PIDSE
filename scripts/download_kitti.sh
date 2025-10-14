#!/bin/bash

# KITTI Data Downloader for PIDSE
# Downloads minimal KITTI data needed for vehicle dynamics learning

echo "🚗 KITTI Data Downloader for PIDSE"
echo "================================="

# Create KITTI data directory
mkdir -p data/kitti
cd data/kitti

echo "📁 Created data/kitti directory"

# Download essential files
echo "📥 Downloading KITTI odometry poses (28MB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
if [ $? -eq 0 ]; then
    echo "✅ Poses downloaded successfully"
    unzip -q data_odometry_poses.zip
    rm data_odometry_poses.zip
    echo "📦 Extracted poses to poses/ directory"
else
    echo "❌ Failed to download poses"
    exit 1
fi

echo "📥 Downloading KITTI calibration files (1MB)..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
if [ $? -eq 0 ]; then
    echo "✅ Calibration downloaded successfully"
    unzip -q data_odometry_calib.zip
    rm data_odometry_calib.zip
    echo "📦 Extracted calibration to sequences/ directory"
else
    echo "❌ Failed to download calibration"
    exit 1
fi

# Optional: Download sequences (large)
echo ""
read -p "🤔 Download sequence images? (y/N) [Adds ~22GB]: " download_images

if [[ $download_images =~ ^[Yy]$ ]]; then
    echo "📥 Downloading sequence images (this will take a while)..."
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
    if [ $? -eq 0 ]; then
        echo "✅ Images downloaded successfully"
        unzip -q data_odometry_color.zip
        rm data_odometry_color.zip
    else
        echo "❌ Failed to download images"
    fi
else
    echo "⏭ Skipping images - using poses only"
fi

cd ../..

echo ""
echo "🎉 KITTI download completed!"
echo "📊 Downloaded data:"
ls -la data/kitti/

echo ""
echo "📁 KITTI structure:"
echo "data/kitti/"
echo "├── poses/          # Ground truth trajectories"
echo "│   ├── 00.txt      # Sequence 00 poses"
echo "│   ├── 01.txt      # Sequence 01 poses"
echo "│   └── ..."
echo "└── sequences/      # Calibration files"
echo "    ├── 00/"
echo "    ├── 01/"
echo "    └── ..."

echo ""
echo "🚀 Ready for PIDSE training!"
echo "   Run: python examples/kitti_example.py"