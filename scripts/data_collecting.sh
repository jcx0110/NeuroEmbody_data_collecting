# ===========================================
# STYLING & HELPER FUNCTIONS
# ===========================================

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get current timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Standardized Logging Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(timestamp) | $1"
}

log_success() {
    echo -e "${GREEN}[OK]  ${NC} $(timestamp) | $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(timestamp) | $1"
}

log_error() {
    echo -e "${RED}[ERR] ${NC} $(timestamp) | $1"
}

# Header function
print_header() {
    local title="$1"
    echo -e "${CYAN}"
    echo "========================================================"
    printf "  %*s\n" $(((${#title} + 54) / 2)) "$title"
    echo "========================================================"
    echo -e "${NC}"
}

# Cleanup function
cleanup() {
    echo ""
    log_warn "Interrupt signal received. Shutting down..."
    
    # Kill data collection client (may be running in foreground, so use pkill)
    log_warn "Terminating Data Collection Client..."
    pkill -TERM -f "run_data_collecting.py" 2>/dev/null
    sleep 1
    # Force kill if still running
    pkill -9 -f "run_data_collecting.py" 2>/dev/null && log_warn "Force killed Data Collection Client"
    
    # Kill robot server if running
    if [ ! -z "$SERVER_PID" ]; then
        log_warn "Killing Robot Server (PID: $SERVER_PID)..."
        # First try graceful termination
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill -9 $SERVER_PID 2>/dev/null
            log_warn "Force killed Robot Server"
        fi
        log_success "Robot Server terminated"
    fi
    
    # Additional cleanup: kill any remaining python processes related to data collection
    # This ensures no orphaned processes remain
    pkill -f "arm_server.py" 2>/dev/null && log_warn "Killed remaining arm_server.py processes"
    pkill -f "run_data_collecting.py" 2>/dev/null && log_warn "Killed remaining run_data_collecting.py processes"
    
    log_info "Cleanup complete. Exiting."
    print_header "End NeuroEmbody Data Collection"
    exit
}

# ===========================================
# MAIN EXECUTION
# ===========================================

# Ensure conda environment is activated (if conda is available)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate NeuroEmbody 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate NeuroEmbody 2>/dev/null || true
fi

# Verify Python path
PYTHON_CMD=$(which python)
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi
log_info "Using Python: $PYTHON_CMD"

trap cleanup SIGINT SIGTERM EXIT

print_header "Start NeuroEmbody Data Collection"

echo "-----------Step 1/2: Launching Robot Server for UR5-----------"

# 1. Launch Robot Server
log_info "Target IP: $ROBOT_IP"

# Start server in background and capture PID
$PYTHON_CMD data_collecting/core/arm_server.py &
SERVER_PID=$!

# Create a new process group for the server to make cleanup easier
if [ ! -z "$SERVER_PID" ]; then
    # Try to create a new process group (may not work if process already started)
    # The important part is we have the PID now
    :
fi

log_info "Server process spawned with PID: $SERVER_PID"
log_info "Waiting 3 seconds for initialization..."

sleep 3

# Verify server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    log_error "Server process died during initialization!"
    exit 1
fi

log_success "Server is running."

echo "-----------Step 2/2: Starting Data Collection Client-----------"

# Using printf for aligned parameter display
printf "   %-15s : %s\n" "Task Name" "$TASK_NAME"
printf "   %-15s : %s\n" "Input Device" "GELLO"
printf "   %-15s : %s\n" "Robot Port" "6001"

echo "" # Empty line for separation

# Start data collection client in foreground (needs keyboard input)
# When user presses Ctrl+C or 'q', this will exit and trigger cleanup
log_info "Starting Data Collection Client..."
$PYTHON_CMD data_collecting/core/run_data_collecting.py 2>/dev/null

CLIENT_EXIT_CODE=$?

if [ $CLIENT_EXIT_CODE -ne 0 ]; then
    log_warn "Data Collection Client exited with code: $CLIENT_EXIT_CODE"
fi

# Script ends, trap triggers cleanup