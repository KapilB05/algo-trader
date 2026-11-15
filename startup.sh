#!/usr/bin/env bash
# ==============================================================================
#  Algo-Trader — Unified CLI Process Controller (All-in-One)
# ==============================================================================
#  Usage:
#    ./startup.sh           Start all services in background (default)
#    ./startup.sh start     Start all services in background
#    ./startup.sh stop      Gracefully stop all services
#    ./startup.sh status    Check running status & view recent logs
#    ./startup.sh restart   Gracefully stop & restart all services
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

# ANSI Color codes
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Ensure working directories
mkdir -p logs pids

# Determine best Python executable
get_python() {
    if [ -f ".venv/Scripts/python.exe" ]; then
        echo ".venv/Scripts/python.exe"
    elif [ -f "venv/Scripts/python.exe" ]; then
        echo "venv/Scripts/python.exe"
    elif [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
    elif [ -f "venv/bin/python" ]; then
        echo "venv/bin/python"
    elif command -v python.exe &>/dev/null; then
        echo "python.exe"
    elif command -v python &>/dev/null && python -c "import uvicorn" &>/dev/null; then
        echo "python"
    elif command -v python3 &>/dev/null && python3 -c "import uvicorn" &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        echo "python3"
    fi
}

PYTHON_CMD=$(get_python)

# Helper: check if PID file belongs to a live process
is_pid_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0 # Running
        fi
    fi
    return 1 # Stopped
}

# ── 1. START ACTION ────────────────────────────────────────────────────────────
do_start() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}   🏛️  ALGO-TRADER: PRODUCTION SYSTEM STARTUP LAUNCHER          ${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "  Using Python  : ${GREEN}$($PYTHON_CMD --version 2>&1)${NC} (${PYTHON_CMD})"
    echo -e "  Root Directory: ${CYAN}${SCRIPT_DIR}${NC}\n"

    # Check for already running services
    local any_running=0
    if is_pid_running "pids/paper_trader.pid"; then
        echo -e "${AMBER}[!] Warning: Paper Trader is already running (PID: $(cat pids/paper_trader.pid))${NC}"
        any_running=1
    fi
    if is_pid_running "pids/auto_learner.pid"; then
        echo -e "${AMBER}[!] Warning: Auto-Learner is already running (PID: $(cat pids/auto_learner.pid))${NC}"
        any_running=1
    fi
    if is_pid_running "pids/dashboard.pid"; then
        echo -e "${AMBER}[!] Warning: Dashboard UI is already running (PID: $(cat pids/dashboard.pid))${NC}"
        any_running=1
    fi

    if [ $any_running -eq 1 ]; then
        echo -e "\n${RED}[ABORT] One or more services are already running.${NC}"
        echo -e "Run ${CYAN}./startup.sh status${NC} or ${CYAN}./startup.sh restart${NC}\n"
        exit 1
    fi

    # Start Paper Trader
    echo -e "[1/3] Starting Live Paper Trader..."
    PYTHONUNBUFFERED=1 nohup "$PYTHON_CMD" run_paper_trader.py </dev/null > logs/paper_trader.out 2> logs/paper_trader.err &
    local paper_pid=$!
    disown $paper_pid 2>/dev/null || true
    echo "$paper_pid" > pids/paper_trader.pid
    echo -e "  ${GREEN}✔ Paper Trader started (PID: ${paper_pid})${NC}"

    # Start Auto-Learner Scheduler
    echo -e "\n[2/3] Starting Auto-Learner Scheduler..."
    PYTHONUNBUFFERED=1 nohup "$PYTHON_CMD" run_auto_learning.py </dev/null > logs/auto_learner.out 2> logs/auto_learner.err &
    local auto_pid=$!
    disown $auto_pid 2>/dev/null || true
    echo "$auto_pid" > pids/auto_learner.pid
    echo -e "  ${GREEN}✔ Auto-Learner Scheduler started (PID: ${auto_pid})${NC}"

    # Start Dashboard Server
    echo -e "\n[3/3] Starting Bloomberg Trading Terminal Dashboard (Port 8080)..."
    sleep 1
    PYTHONUNBUFFERED=1 nohup "$PYTHON_CMD" run_dashboard.py </dev/null > logs/dashboard.out 2> logs/dashboard.err &
    local dash_pid=$!
    disown $dash_pid 2>/dev/null || true
    echo "$dash_pid" > pids/dashboard.pid
    echo -e "  ${GREEN}✔ Dashboard Server started (PID: ${dash_pid})${NC}"
    echo -e "    URL: ${CYAN}http://localhost:8080/${NC}"

    echo -e "\n${GREEN}================================================================${NC}"
    echo -e "${GREEN}   ✨  ALL 3 SERVICES SUCCESSFULLY STARTED IN BACKGROUND       ${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo -e "  • Check Status : ${CYAN}./startup.sh status${NC}"
    echo -e "  • Stop System  : ${CYAN}./startup.sh stop${NC}"
    echo -e "  • Restart All  : ${CYAN}./startup.sh restart${NC}"
    echo -e "  • Live UI      : ${CYAN}http://localhost:8080/${NC}"
    echo -e "================================================================\n"
}

# ── 2. STOP ACTION ─────────────────────────────────────────────────────────────
stop_one_service() {
    local name="$1"
    local pid_file="$2"

    if [ ! -f "$pid_file" ]; then
        echo -e "  • ${name}: ${NC}Not running"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ]; then
        rm -f "$pid_file"
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "  • Stopping ${CYAN}${name}${NC} (PID: ${pid}) with SIGTERM..."
        kill -15 "$pid" 2>/dev/null || true

        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 4 ]; do
            sleep 0.5
            count=$((count + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            echo -e "    ${RED}⚠ Force killing ${name} (PID: ${pid}) with SIGKILL...${NC}"
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo -e "    ${GREEN}✔ ${name} stopped successfully.${NC}"
    else
        echo -e "  • ${name}: ${NC}Process (PID ${pid}) was already stopped."
    fi

    rm -f "$pid_file"
}

do_stop() {
    echo -e "${AMBER}================================================================${NC}"
    echo -e "${AMBER}   🛑  ALGO-TRADER: INITIATING GRACEFUL SERVICE SHUTDOWN        ${NC}"
    echo -e "${AMBER}================================================================${NC}"

    stop_one_service "Live Paper Trader" "pids/paper_trader.pid"
    stop_one_service "Auto-Learner Scheduler" "pids/auto_learner.pid"
    stop_one_service "Dashboard UI (Port 8080)" "pids/dashboard.pid"

    # Clean port 8080 on Windows / Linux
    if [ -n "$PYTHON_CMD" ]; then
        "$PYTHON_CMD" -c "import subprocess, sys; subprocess.run('powershell -Command \"(Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique | ForEach-Object { if ($_ -gt 0) { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }\"', shell=True) if sys.platform == 'win32' else None" 2>/dev/null || true
    fi
    if command -v fuser &>/dev/null; then
        fuser -k 8080/tcp 2>/dev/null || true
    fi

    echo -e "\n${GREEN}================================================================${NC}"
    echo -e "${GREEN}   ✔  SHUTDOWN COMPLETE: All services halted and cleaned up.   ${NC}"
    echo -e "${GREEN}================================================================\n"
}

# ── 3. STATUS ACTION ───────────────────────────────────────────────────────────
check_service_status() {
    local name="$1"
    local pid_file="$2"

    printf "  %-30s : " "${name}"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}${BOLD}RUNNING${NC} (PID: ${CYAN}${pid}${NC})"
        else
            echo -e "${RED}${BOLD}STOPPED${NC} (Stale PID: ${pid})"
        fi
    else
        echo -e "${AMBER}${BOLD}STOPPED${NC}"
    fi
}

show_log_tail() {
    local title="$1"
    local log_file="$2"
    echo -e "\n${BOLD}▶ ${title} (${log_file})${NC}"
    if [ -f "$log_file" ] && [ -s "$log_file" ]; then
        tail -n 5 "$log_file" | sed 's/^/    /'
    else
        echo -e "    ${AMBER}(Log file empty or not yet generated)${NC}"
    fi
}

do_status() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}   🩺  ALGO-TRADER: SYSTEM HEALTH & SERVICE STATUS             ${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "  Checked At: $(date)\n"

    echo -e "${BOLD}--- Service Process Health ---${NC}"
    check_service_status "1. Live Paper Trader" "pids/paper_trader.pid"
    check_service_status "2. Auto-Learner Scheduler" "pids/auto_learner.pid"
    check_service_status "3. Bloomberg Dashboard UI" "pids/dashboard.pid"

    echo -e "\n${BOLD}--- Dashboard URL ---${NC}"
    echo -e "  URL: ${CYAN}http://localhost:8080/${NC}"

    echo -e "\n${CYAN}================================================================${NC}"
    echo -e "${CYAN}   📜  RECENT SERVICE ACTIVITY (LAST 5 LOG LINES)               ${NC}"
    echo -e "${CYAN}================================================================${NC}"

    show_log_tail "Paper Trader Output" "logs/paper_trader.out"
    show_log_tail "Auto-Learner Log" "logs/auto_learning.log"
    show_log_tail "Dashboard Server Output" "logs/dashboard.out"
    echo -e "\n${CYAN}================================================================${NC}\n"
}

# ── 4. RESTART ACTION ──────────────────────────────────────────────────────────
do_restart() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}   🔄  ALGO-TRADER: RESTARTING ALL BACKGROUND SERVICES          ${NC}"
    echo -e "${CYAN}================================================================${NC}\n"
    do_stop
    sleep 1
    do_start
}

# ── CLI ROUTER ────────────────────────────────────────────────────────────────
RAW_ACTION="$1"
# Strip any trailing \r or whitespace
CLEAN_ACTION=$(echo "$RAW_ACTION" | tr -d '\r\n ' | tr '[:upper:]' '[:lower:]')

if [ -z "$CLEAN_ACTION" ]; then
    CLEAN_ACTION="start"
fi

case "$CLEAN_ACTION" in
    start|"")
        do_start
        ;;
    stop)
        do_stop
        ;;
    status)
        do_status
        ;;
    restart)
        do_restart
        ;;
    help|--help|-h)
        echo "Usage: ./startup.sh [start|stop|status|restart]"
        ;;
    *)
        echo -e "${RED}Unknown action: '$1'${NC}"
        echo "Usage: ./startup.sh [start|stop|status|restart]"
        exit 1
        ;;
esac