import h5py
import numpy as np
import os
from pathlib import Path

def discover_and_validate(file_path):
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 动态发现所有 Dataset
            all_datasets = []
            
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append((name, obj))
            
            f.visititems(find_datasets)

            if not all_datasets:
                print("[-] No datasets found in this file.")
                return

            print(f"[+] Discovered {len(all_datasets)} datasets:")
            
            # 2. 打印结构并检查基本完整性
            data_info = {}
            for name, ds in all_datasets:
                shape = ds.shape
                dtype = ds.dtype
                # 读取一小块数据检查是否全零或含有 NaN
                sample = ds[:] if ds.size < 1000000 else ds[0] # 大文件只取首帧
                
                is_all_zero = np.all(sample == 0)
                has_nan = np.any(np.isnan(sample)) if np.issubdtype(dtype, np.number) else False
                
                print(f"  - {name:25} | Shape: {str(shape):15} | Type: {dtype}")
                if is_all_zero: print(f"    \033[93m[!] Warning: Dataset '{name}' is all zeros.\033[0m")
                if has_nan:     print(f"    \033[91m[!] Error: Dataset '{name}' contains NaN values.\033[0m")
                
                # 记录首维长度用于对齐校验
                if len(shape) > 0:
                    data_info[name] = shape[0]

            # 3. 动态对齐校验 (Frame Alignment)
            print(f"\n[+] Synchronization Check:")
            unique_lengths = {}
            for name, length in data_info.items():
                if length not in unique_lengths:
                    unique_lengths[length] = []
                unique_lengths[length].append(name)

            if len(unique_lengths) > 1:
                print(f"  \033[91m[!] Mismatch: Found {len(unique_lengths)} different frame counts:\033[0m")
                for length, names in unique_lengths.items():
                    print(f"    - Length {length}: {', '.join(names)}")
            else:
                print(f"  \033[92m[✓] Alignment: All {len(data_info)} sequential datasets have {next(iter(unique_lengths))} frames.\033[0m")

            # 4. 读取 Attributes (元数据)
            if f.attrs:
                print(f"\n[+] Metadata Attributes:")
                for attr_key, attr_val in f.attrs.items():
                    print(f"  - {attr_key}: {attr_val}")

    except OSError as e:
        print(f"\033[91m[CRITICAL] Could not open file: {e}\033[0m")
    except Exception as e:
        print(f"\033[91m[ERROR] Analysis failed: {e}\033[0m")

# 使用示例
if __name__ == "__main__":
    # 指向你最新的 episode 文件
    target_file = os.path.expanduser("/home/chenxi/chenxi/NeuroEmbody/NeuroEmbody_data_collecting/output/seasoning_process/episode_000.h5")
    
    if os.path.exists(target_file):
        discover_and_validate(target_file)
    else:
        print(f"File not found: {target_file}")