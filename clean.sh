#!/bin/bash

$HOME=/es01/home/lvxg/vdb/VectorDB
# Define cleanup functions
clean_log_files() {
    echo "Cleaning log files..."
    rm -f $HOME/benchmarks/*/result/log.csv 
    rm -f $HOME/benchmarks/*/result/processed_log.csv 
    echo "Log files cleaned."
}

clean_output_files() {
    echo "Cleaning output files..."
    rm -f $HOME/log/*.out $HOME/log/*.err
    echo "Output files cleaned."
}

cancel_jobs() {
    echo "Cancelling all jobs..."
    scancel -u lvxg
    echo "Jobs cancelled."
}

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --logs       Clean log files (log.csv and processed_log.csv)"
    echo "  --outputs    Clean output files (.out and .err)"
    echo "  --jobs       Cancel all jobs for user lvxg"
    echo "  --all        Perform all cleanup operations"
    echo "  --help       Show this help message"
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logs)
            clean_log_files
            shift
            ;;
        --outputs)
            clean_output_files
            shift
            ;;
        --jobs)
            cancel_jobs
            shift
            ;;
        --all)
            cancel_jobs
            clean_log_files
            clean_output_files
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Cleanup operations completed."

# #!/bin/bash
# rm -f ./log/*.out ./log/*.err
