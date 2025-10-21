#!/bin/bash
# Real-time GPU and Training Monitor - Combined version
# Monitors GPU status, training processes, and training logs

echo "=========================================="
echo "Training & GPU Monitor"
echo "=========================================="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "GPU Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    # Show GPU utilization, memory, temperature, and power
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s (%s)\n  Utilization: GPU=%s%% MEM=%s%%\n  Memory: %s/%s MB (%.1f%%)\n  Temp: %s°C | Power: %s/%s W\n\n", 
                     $1, $2, $3, $4, $5, $6, ($5/$6)*100, $7, $8, $9}'
    
    echo "=========================================="
    echo "GPU Processes"
    echo "=========================================="
    
    # Show GPU processes with memory usage
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | \
        awk -F', ' '{printf "  PID %s: %s (GPU Mem: %s)\n", $1, $2, $3}' || echo "  No active GPU processes"
    
    echo ""
    echo "=========================================="
    echo "Training Process Info"
    echo "=========================================="
    
    # Find Python training processes
    TRAIN_PROCS=$(ps aux | grep -E "python.*train.py" | grep -v grep | head -3)
    if [ -z "$TRAIN_PROCS" ]; then
        echo "  No training processes found"
    else
        echo "$TRAIN_PROCS" | awk '{printf "PID: %s | CPU: %s%% | MEM: %s%% | CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    fi
    
    echo ""
    echo "=========================================="
    echo "Recent Training Logs (last 5 lines)"
    echo "=========================================="
    
    # Show recent training output if log file exists
    if [ -f "outputs/logs/train.log" ]; then
        tail -5 outputs/logs/train.log
    else
        echo "No log file found at outputs/logs/train.log"
        echo "Training output should be visible in the terminal"
    fi
    
    echo ""
    echo "Refresh interval: 2 seconds"
    
    sleep 2
done
