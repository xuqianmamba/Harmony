#!/bin/bash

# Define the base directory
VDB_DIR="/es01/home/lvxg/vdb/VectorDB"

# Define cleanup functions
clean_log_files() {
    echo "Cleaning log files..."
    rm -f "$VDB_DIR"/benchmarks/*/result/*.csv 
    rm -f "$VDB_DIR"/benchmarks/*/result/processed_log.csv 
    echo "Log files cleaned."
}

clean_output_files() {
    echo "Cleaning output files..."
    rm -f "$VDB_DIR"/log/*.out "$VDB_DIR"/log/*.err
    echo "Output files cleaned."
}

clean_index_and_result_files() {
    echo "Cleaning index/ and result/ files..."
    # 删除 index 下的文件
    find "$VDB_DIR"/benchmarks/*/index -type f -print -delete
    # 删除 result 下的文件
    find "$VDB_DIR"/benchmarks/*/result -type f -print -delete
    echo "Index and result files cleaned."
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
    echo "  --indexes    Clean all files in */index and */result directories"
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
        --indexes)
            clean_index_and_result_files
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
            clean_index_and_result_files
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
