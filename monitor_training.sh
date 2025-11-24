#!/bin/bash
# Monitor KITTI training progress

echo "📊 PIDSE KITTI Training Monitor"
echo "================================"
echo ""

# Check if process is running
if pgrep -f "kitti_full_training.py" > /dev/null; then
    echo "✅ Training process is RUNNING"
else
    echo "⏹️  Training process is NOT running"
fi

echo ""
echo "📄 Latest Training Output:"
echo "----------------------------"
tail -n 30 kitti_training_log.txt

echo ""
echo "----------------------------"
echo "💡 Commands:"
echo "   Watch live: tail -f kitti_training_log.txt"
echo "   Kill training: pkill -f kitti_full_training"
echo "   Check MLflow: venv/bin/mlflow ui"
