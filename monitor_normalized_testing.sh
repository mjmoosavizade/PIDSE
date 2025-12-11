#!/bin/bash
# Monitor normalized hypothesis testing progress

LOG_FILE="experiments/normalized_hypothesis_testing_log.txt"

echo "================================================================================"
echo "Normalized Physics Hypothesis Testing Monitor"
echo "================================================================================"
echo "Current time: $(date)"
echo ""

# Check if process is running
if pgrep -f "test_normalized_hypotheses.py" > /dev/null; then
    PID=$(pgrep -f 'test_normalized_hypotheses.py' | head -1)
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    TIME=$(ps -p $PID -o time= | tr -d ' ')
    echo "✅ Process RUNNING (PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%, Time: $TIME)"
else
    echo "❌ Process NOT RUNNING"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "Recent Progress (last 40 lines):"
echo "--------------------------------------------------------------------------------"
tail -40 "$LOG_FILE"

echo ""
echo "--------------------------------------------------------------------------------"
echo "Summary:"
echo "--------------------------------------------------------------------------------"

# Count completed epochs across all configs
EPOCHS_DONE=$(grep "^Epoch" "$LOG_FILE" 2>/dev/null | wc -l)
echo "  Total epoch lines: $EPOCHS_DONE"

# Check for best results
BEST_ATE=$(grep "New best ATE" "$LOG_FILE" 2>/dev/null | tail -1)
if [ ! -z "$BEST_ATE" ]; then
    echo "  Latest: $BEST_ATE"
fi

# Count hypotheses completed
COMPLETED=$(grep -c "✅ Completed" "$LOG_FILE" 2>/dev/null || echo "0")
echo "  Configurations completed: $COMPLETED / 15"

# Check for errors
ERRORS=$(grep -c "❌ Failed" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$ERRORS" -gt 0 ]; then
    echo "  ⚠️  Errors encountered: $ERRORS"
fi

# Current hypothesis
CURRENT=$(grep "# HYPOTHESIS:" "$LOG_FILE" | tail -1)
if [ ! -z "$CURRENT" ]; then
    echo "  Current: $CURRENT"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  Check status: bash monitor_normalized_testing.sh"
echo "================================================================================"
