# get_research_df Refactoring Summary

## Overview
The `get_research_df` function in `reaserch_utils/reaserch_dataframe.py` has been refactored to separate video processing into two distinct phases with minimal logic changes.

## Changes Made

### New Architecture
1. **Phase 1**: `_process_videos_phase()` - Video processing (results.npy auto-saved)
2. **Phase 2**: `_cycles_and_dataframe_phase()` - Cycle extraction and DataFrame generation
3. **Main Function**: `get_research_df()` - Orchestrates both phases

### Key Features
- **Backward Compatible**: Default behavior unchanged (`skip_phase_1=False`)
- **Uses Existing Caching**: Leverages `results.npy` already saved in output_dir
- **Phase Independence**: Phase 2 can run independently by loading from results.npy
- **Minimal Changes**: Existing code logic preserved exactly

## Usage

### Mode 1: Full Processing (Default)
```python
df, exp_df = get_research_df(
    exp=exp,
    exp_df=exp_df,
    pipe=pipe,
    experiment_outputs=outputs_dir
)
```

### Mode 2: Process with Output Saving
```python
# First run: Process videos and save results.npy
df, exp_df = get_research_df(
    exp=exp,
    exp_df=exp_df,
    pipe=pipe,
    experiment_outputs=outputs_dir,
    generate_output=True  # Saves results.npy in output_dir
)
```

### Mode 3: Load from Cached Results
```python
# Later runs: Skip processing, load results.npy
df, exp_df = get_research_df(
    exp=exp,
    exp_df=exp_df,
    pipe=pipe,
    experiment_outputs=outputs_dir,
    skip_phase_1=True  # Load results.npy from output_dir
)
```

## Implementation Details

### _process_videos_phase()
- Processes all videos sequentially with progress bar
- Adds trial info to Results objects
- Leverages pipe.process() which auto-saves results.npy
- Returns: (results_dict, failed_main, failed_cycles, exp_df)

### _cycles_and_dataframe_phase()
- Extracts cycles from Results objects
- Processes dynamics for sagittal plane videos
- Generates pandas DataFrame from cycles
- Returns: (df, exp_df, failed_cycles)

## File Structure
- **Input**: Results objects (from Phase 1 or results.npy)
- **Cache Format**: results.npy in `output_dir/club/swimmer/video_name/`
- **Output**: DataFrame and updated exp_df

## Benefits
1. Faster iteration: Run Phase 2 multiple times without reprocessing videos
2. Flexible processing: Skip expensive video processing when results cached
3. Backward compatible: Existing code works without changes
4. Clear separation of concerns
5. Reuses existing caching: Leverages results.npy already saved by pipe.process()

## Technical Implementation Details

### Data Flow Diagram
```
Phase 1: Video Processing
Video Files → pipe.process() → results.npy saved to output_dir → Results in memory

Phase 2: Cycles & DataFrame Generation
Results Objects → cycles extraction → to_table() → concat → DataFrame
```

### Cache File Format
- Format: NumPy `.npy` files with `allow_pickle=True`
- Location: `{experiment_outputs}/{club}/{swimmer}/{video_name}/results.npy`
- Created by: `pipe.process()` when output_dir is specified

### Files Modified
- `reaserch_utils/reaserch_dataframe.py`: Refactored main function

### Existing Callers (No Changes Required)
- `clustering_swimmers_new/a_generate_dataset.py`
- `average_swimmer/dataset_creation/a_create_average_cycle_table.py`
- Multiple notebooks - all backward compatible
