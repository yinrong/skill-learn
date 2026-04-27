#!/bin/bash
# Watchdog: if a batch launcher dies before writing all results,
# launch the Python monitor as a fallback.
# Run: nohup bash watchdog.sh > history-route2.1.1/logs/watchdog.log 2>&1 &

cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Wait until batch6 monitor confirms it's done, then watch batch7-11
declare -A LAUNCHER_PID
LAUNCHER_PID[7]=115871
LAUNCHER_PID[8]=116058
LAUNCHER_PID[9]=118349
LAUNCHER_PID[10]=119507
LAUNCHER_PID[11]=120307

declare -A BATCH_EXPS
BATCH_EXPS[7]="expZZ expAAA expBBB expCCC"
BATCH_EXPS[8]="expGGG expHHH"
BATCH_EXPS[9]="expIII expJJJ expKKK expLLL expMMM"
BATCH_EXPS[10]="expNNN expOOO expPPP expQQQ expSSS"
BATCH_EXPS[11]="expTTT expUUU expVVV expWWW"

declare -A MONITOR_LAUNCHED

while true; do
    for batch in 7 8 9 10 11; do
        [ "${MONITOR_LAUNCHED[$batch]}" = "1" ] && continue

        # Check if any training log exists for this batch (means launcher started training)
        exps="${BATCH_EXPS[$batch]}"
        first_exp=$(echo $exps | cut -d' ' -f1)
        train_log="$HISTDIR/logs/${first_exp}_train.log"

        # If training started but launcher is dead → fallback to Python monitor
        if [ -f "$train_log" ]; then
            # Skip if a dedicated monitor is already handling this batch
            if pgrep -f "monitor_batch${batch}.py" > /dev/null 2>&1; then
                MONITOR_LAUNCHED[$batch]="1"
                continue
            fi
            pid="${LAUNCHER_PID[$batch]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                # Check if all results already exist
                all_done=true
                for exp in $exps; do
                    [ ! -f "$HISTDIR/results/${exp}.json" ] && all_done=false && break
                done

                if [ "$all_done" = "false" ]; then
                    # Don't start if a Python monitor is already running for this batch
                    if pgrep -f "monitor_batch${batch}.py\|monitor_batch.py --batch $batch" > /dev/null 2>&1; then
                        log "Batch $batch: Python monitor already running, no action"
                        MONITOR_LAUNCHED[$batch]="1"
                    else
                        log "⚠ Batch $batch launcher (PID=$pid) died! Starting Python monitor..."
                        MONITOR_LAUNCHED[$batch]="1"
                        nohup python3 monitor_batch.py --batch $batch \
                            > "$HISTDIR/logs/batch${batch}_pymonitor.log" 2>&1 &
                        log "   Python monitor started (PID=$!, batch=$batch)"
                    fi
                else
                    log "Batch $batch: all results exist, no action needed"
                    MONITOR_LAUNCHED[$batch]="1"
                fi
            fi
        fi
    done

    # Exit when all batches through 11 are complete (results exist)
    all_complete=true
    for exp in expZZ expAAA expBBB expCCC expGGG expHHH expIII expJJJ expKKK expLLL expMMM; do
        [ ! -f "$HISTDIR/results/${exp}.json" ] && all_complete=false && break
    done
    if [ "$all_complete" = "true" ]; then
        log "All watched batches complete. Watchdog exiting."
        exit 0
    fi

    sleep 120
done
