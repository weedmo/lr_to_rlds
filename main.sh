#!/bin/bash
# main.sh - Interactive entry point for LeRobot to RLDS converter
#
# Usage: ./main.sh
#
# Provides an interactive menu for:
# - Listing RLDS datasets in data/
# - Discovering LeRobot dataset structure
# - Visualizing RLDS datasets (episodes, plots, frames)
# - Converting LeRobot datasets to RLDS format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Activate virtual environment if it exists
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "${VENV_DIR}/bin/activate"
    else
        echo -e "${YELLOW}Warning: Virtual environment not found at ${VENV_DIR}${NC}"
        echo -e "${YELLOW}Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'${NC}"
    fi
}

# Print header
print_header() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   LeRobot to RLDS Converter${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Print main menu
print_main_menu() {
    echo -e "${GREEN}Main Menu:${NC}"
    echo ""
    echo "  1) List RLDS datasets in data/"
    echo "  2) Discover LeRobot dataset (source)"
    echo "  3) Visualize RLDS dataset"
    echo "  4) Convert LeRobot to RLDS"
    echo ""
    echo "  h) Help"
    echo "  q) Quit"
    echo ""
}

# Print visualize menu
print_visualize_menu() {
    echo -e "${GREEN}Visualize RLDS Dataset:${NC}"
    echo ""
    echo "  1) List episodes"
    echo "  2) Show dataset info"
    echo "  3) Plot state/action"
    echo "  4) View frames"
    echo "  5) List cameras"
    echo "  6) Export frames as images"
    echo ""
    echo "  b) Back to main menu"
    echo ""
}

# Get RLDS dataset path from data/
get_rlds_dataset_path() {
    local prompt="${1:-Enter RLDS dataset path}"

    # Check if data/ exists and has subdirectories
    if [ -d "$DATA_DIR" ]; then
        local datasets=($(find "$DATA_DIR" -maxdepth 2 -name "dataset_info.json" -exec dirname {} \; 2>/dev/null | head -10))
        if [ ${#datasets[@]} -gt 0 ]; then
            echo -e "${BLUE}Available RLDS datasets:${NC}"
            for d in "${datasets[@]}"; do
                local rel_path="${d#$DATA_DIR/}"
                echo "  - $rel_path"
            done
            echo ""
        fi
    fi

    read -p "$prompt (or relative to data/): " path

    # If path doesn't exist, try prepending data/
    if [ ! -d "$path" ] && [ -d "${DATA_DIR}/${path}" ]; then
        path="${DATA_DIR}/${path}"
    fi

    echo "$path"
}

# Get LeRobot source dataset path
get_lerobot_dataset_path() {
    local prompt="${1:-Enter LeRobot dataset path}"
    read -p "$prompt: " path
    echo "$path"
}

# List datasets
do_list_datasets() {
    echo ""
    echo -e "${BLUE}Listing RLDS datasets...${NC}"
    echo ""
    lerobot-to-rlds list-datasets --data-dir "$DATA_DIR" 2>/dev/null || {
        echo -e "${YELLOW}No RLDS datasets found or data/ directory doesn't exist.${NC}"
    }
    echo ""
    read -p "Press Enter to continue..."
}

# Discover dataset
do_discover() {
    echo ""
    local path=$(get_lerobot_dataset_path "Enter LeRobot source dataset path")

    if [ -z "$path" ] || [ ! -d "$path" ]; then
        echo -e "${RED}Invalid path.${NC}"
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    echo -e "${BLUE}Discovering LeRobot dataset: ${path}${NC}"
    echo ""
    lerobot-to-rlds discover "$path"
    echo ""
    read -p "Press Enter to continue..."
}

# Visualize submenu
do_visualize() {
    while true; do
        print_header
        print_visualize_menu

        read -p "Select option: " choice

        case $choice in
            1)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    echo ""
                    lerobot-to-rlds visualize list "$path"
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    echo ""
                    lerobot-to-rlds visualize info "$path"
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    read -p "Episode index [0]: " episode
                    episode=${episode:-0}
                    read -p "Plot type (state/action/both) [both]: " ptype
                    ptype=${ptype:-both}
                    read -p "Save to file (leave empty to display): " save_path

                    echo ""
                    if [ -n "$save_path" ]; then
                        lerobot-to-rlds visualize plot "$path" -e "$episode" -t "$ptype" -s "$save_path" --no-show
                    else
                        lerobot-to-rlds visualize plot "$path" -e "$episode" -t "$ptype"
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    read -p "Episode index [0]: " episode
                    episode=${episode:-0}
                    read -p "Step index (or comma-separated for grid) [0]: " steps
                    steps=${steps:-0}
                    read -p "Camera (image/wrist_image) [image]: " camera
                    camera=${camera:-image}

                    echo ""
                    if [[ "$steps" == *","* ]]; then
                        lerobot-to-rlds visualize frames "$path" -e "$episode" --steps "$steps" -c "$camera"
                    else
                        lerobot-to-rlds visualize frames "$path" -e "$episode" -s "$steps" -c "$camera"
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    read -p "Episode index [0]: " episode
                    episode=${episode:-0}
                    echo ""
                    lerobot-to-rlds visualize cameras "$path" -e "$episode"
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                local path=$(get_rlds_dataset_path "Enter RLDS dataset path")
                if [ -n "$path" ] && [ -d "$path" ]; then
                    read -p "Episode index [0]: " episode
                    episode=${episode:-0}
                    read -p "Output directory: " output_dir
                    read -p "Camera (image/wrist_image) [image]: " camera
                    camera=${camera:-image}
                    read -p "Start step (leave empty for all): " start
                    read -p "End step (leave empty for all): " end

                    if [ -n "$output_dir" ]; then
                        echo ""
                        cmd="lerobot-to-rlds visualize export-frames \"$path\" \"$output_dir\" -e $episode -c \"$camera\""
                        [ -n "$start" ] && cmd="$cmd --start $start"
                        [ -n "$end" ] && cmd="$cmd --end $end"
                        eval $cmd
                    else
                        echo -e "${RED}Output directory required.${NC}"
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            b|B)
                return
                ;;
            *)
                echo -e "${RED}Invalid option.${NC}"
                sleep 1
                ;;
        esac
    done
}

# Convert dataset
do_convert() {
    echo ""
    local path=$(get_lerobot_dataset_path "Enter LeRobot source dataset path to convert")

    if [ -z "$path" ] || [ ! -d "$path" ]; then
        echo -e "${RED}Invalid path.${NC}"
        read -p "Press Enter to continue..."
        return
    fi

    read -p "Custom output name (leave empty for auto): " name
    read -p "Output directory (leave empty for data/<name>/): " output
    read -p "Format (oxe/legacy) [oxe]: " format
    format=${format:-oxe}
    read -p "Resume from checkpoint? (y/N): " resume

    echo ""
    echo -e "${BLUE}Converting dataset...${NC}"
    echo ""

    cmd="lerobot-to-rlds convert \"$path\" -f $format"
    [ -n "$name" ] && cmd="$cmd -n \"$name\""
    [ -n "$output" ] && cmd="$cmd -o \"$output\""
    [ "$resume" = "y" ] || [ "$resume" = "Y" ] && cmd="$cmd --resume"

    eval $cmd

    echo ""
    read -p "Press Enter to continue..."
}

# Show help
do_help() {
    echo ""
    echo -e "${GREEN}LeRobot to RLDS Converter Help${NC}"
    echo ""
    echo "This tool converts LeRobot datasets (v2.1/v3.0) to RLDS format"
    echo "for use with OpenVLA, OXE, and other RL frameworks."
    echo ""
    echo -e "${BLUE}Workflow:${NC}"
    echo "  1. Discover: Analyze a LeRobot source dataset"
    echo "  2. Convert:  Convert LeRobot -> RLDS (saves to data/)"
    echo "  3. Visualize: View the converted RLDS dataset"
    echo ""
    echo -e "${BLUE}Quick Commands:${NC}"
    echo "  lerobot-to-rlds discover <lerobot_path>   # Analyze source"
    echo "  lerobot-to-rlds convert <lerobot_path>    # Convert to RLDS"
    echo "  lerobot-to-rlds visualize list <rlds_path>   # List episodes"
    echo "  lerobot-to-rlds visualize plot <rlds_path>   # Plot state/action"
    echo "  lerobot-to-rlds visualize frames <rlds_path> # View images"
    echo ""
    echo -e "${BLUE}Default Output:${NC}"
    echo "  Converted datasets are saved to data/<folder_name>/"
    echo "  Use -o to specify a custom output directory"
    echo "  Use -n to specify a custom folder name"
    echo ""
    echo -e "${BLUE}Formats:${NC}"
    echo "  oxe (default) - OXE/OpenVLA compatible, loadable with tfds.load()"
    echo "  legacy        - Custom format for other pipelines"
    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
main() {
    activate_venv

    while true; do
        print_header
        print_main_menu

        read -p "Select option: " choice

        case $choice in
            1)
                do_list_datasets
                ;;
            2)
                do_discover
                ;;
            3)
                do_visualize
                ;;
            4)
                do_convert
                ;;
            h|H)
                do_help
                ;;
            q|Q)
                echo ""
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option.${NC}"
                sleep 1
                ;;
        esac
    done
}

# Run main
main
