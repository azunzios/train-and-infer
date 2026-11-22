```python
# @title Inferensi Video
video_path = "/content/video_downloaded.mp4" # @param {type:"string"}
output_dir = "/content/results" # @param {type:"string"}
start_time = "0" # @param {type:"string"}
end_time = "" # @param {type:"string"}
profile = "balanced" # @param ["balanced", "fast", "fastest"]
batch_size = 8 # @param [1, 2, 4, 8, 16, 32] {type:"raw"}
stride_sec = 0.5 # @param {type:"number"}
scan_mode = "full" # @param ["full", "adaptive"]
gate_ratio = 0.5 # @param {type:"number"}
refine_padding_sec = 2.0 # @param {type:"number"}
smooth_sec = 1.0 # @param {type:"number"}
merge_gap_sec = 2.0 # @param {type:"number"}
min_duration_sec = 1.0 # @param {type:"number"}
low_ratio = 0.6 # @param {type:"number"}
aggregation = "mean" # @param ["mean", "max"]

!python infer.py --video "{video_path}" --ckpt /content/train-and-infer/model.pt --out-dir "{output_dir}" --profile {profile} --batch {batch_size} --stride {stride_sec} --scan {scan_mode} --gate-ratio {gate_ratio} --refine-padding {refine_padding_sec} --smooth {smooth_sec} --merge-gap {merge_gap_sec} --min-dur {min_duration_sec} --low-ratio {low_ratio} --start-time "{start_time}" --end-time "{end_time}" --agg {aggregation}
```
