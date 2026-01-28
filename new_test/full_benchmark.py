import torch
import time
import json
import os
import threading
import subprocess
import datetime
import math
import torch.multiprocessing as mp

# --- Compatibility Monkeypatch ---
import torchrl.objectives.ppo as ppo
try:
    original_init = ppo.ClipPPOLoss.__init__
    def patched_init(self, *args, **kwargs):
        if "critic_coef" in kwargs: kwargs["critic_coeff"] = kwargs.pop("critic_coef")
        if "entropy_coef" in kwargs: kwargs["entropy_coeff"] = kwargs.pop("entropy_coef")
        return original_init(self, *args, **kwargs)
    ppo.ClipPPOLoss.__init__ = patched_init
except:
    pass
# ---------------------------------

class GPUStressBenchmark:
    def __init__(self, output_file=None):
        if output_file is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            device_name = torch.cuda.get_device_name(0).replace(" ", "_").lower() if torch.cuda.is_available() else "cpu"
            self.output_file = f"/workspace/marl_benchmark/new_test/{device_name}_{ts}.json"
        else:
            self.output_file = output_file
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            "timestamp": str(datetime.datetime.now()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "system": {},
            "low_level": {},
            "rl_stress": {},
            "llm_stress": {}
        }
        self.stop_monitoring = False
        self.peak_mem = 0
        self.peak_util = 0

    def _monitor_resources(self):
        """Background thread to capture peak GPU stats"""
        while not self.stop_monitoring:
            try:
                # Parsing nvidia-smi is safer than requiring pynvml
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], 
                    encoding="utf-8"
                )
                util, mem = map(int, result.strip().split(","))
                self.peak_util = max(self.peak_util, util)
                self.peak_mem = max(self.peak_mem, mem)
            except:
                pass
            time.sleep(0.1)

    def run_low_level(self):
        print("\n🔹 Running Low-Level Compute & Bandwidth Tests...")
        size = 8192 # Large enough for saturation
        
        # 1. FP32 Matrix Mul
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        duration = time.time() - start
        ops = 2 * (size ** 3) * 20 # 2 ops per MAC
        tflops = (ops / duration) / 1e12
        self.results["low_level"]["fp32_tflops"] = tflops
        print(f"   FP32 TFLOPS: {tflops:.2f}")

        # 2. BF16 Tensor Core check
        try:
            a_bf16 = a.to(torch.bfloat16)
            b_bf16 = b.to(torch.bfloat16)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50): # More iter for faster ops
                _ = torch.matmul(a_bf16, b_bf16)
            torch.cuda.synchronize()
            duration = time.time() - start
            ops = 2 * (size ** 3) * 50
            tflops_bf16 = (ops / duration) / 1e12
            self.results["low_level"]["bf16_tflops"] = tflops_bf16
            print(f"   BF16 TFLOPS: {tflops_bf16:.2f}")
        except Exception as e:
            print(f"   BF16 Test Failed: {e}")

        # 3. Bandwidth (Host <-> Device)
        # 1GB Transfer
        t_size = 256 * 1024 * 1024 # floats
        cpu_tensor = torch.randn(t_size)
        
        torch.cuda.synchronize()
        start = time.time()
        gpu_tensor = cpu_tensor.to(self.device)
        torch.cuda.synchronize()
        h2d = (1.0) / (time.time() - start) # GB/s
        
        torch.cuda.synchronize()
        start = time.time()
        _ = gpu_tensor.to("cpu")
        torch.cuda.synchronize()
        d2h = (1.0) / (time.time() - start)
        
        self.results["low_level"]["pcie_h2d_gbps"] = h2d
        self.results["low_level"]["pcie_d2h_gbps"] = d2h
        print(f"   PCIe H2D: {h2d:.2f} GB/s | D2H: {d2h:.2f} GB/s")

        del a, b, cpu_tensor, gpu_tensor
        torch.cuda.empty_cache()

    def _save_results(self):
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=4)

    def run_rl_stress(self):
        print("\n🔹 Running RL Vectorization Stress (VMAS)...")
        # Reuse the high-load setting from previous test
        try:
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.environments.vmas.common import VmasTask as VmasTaskClass
            from benchmarl.algorithms import MappoConfig
            from benchmarl.models import MlpConfig
            
            n_envs = 8192 # Push to peak
            
            exp_config = ExperimentConfig.get_from_yaml()
            exp_config.sampling_device = str(self.device)
            exp_config.train_device = str(self.device)
            exp_config.max_n_iters = 5
            exp_config.loggers = []
            exp_config.evaluation = False
            
            # Use Medium Model for balance
            model_config = MlpConfig(
                num_cells=[512, 512], 
                layer_class=torch.nn.Linear,
                activation_class=torch.nn.ReLU
            )
            
            task = VmasTaskClass.NAVIGATION.get_from_yaml()
            task._config = {"num_envs": n_envs, "max_steps": 100}

            experiment = Experiment(
                task=task,
                algorithm_config=MappoConfig.get_from_yaml(),
                model_config=model_config,
                critic_model_config=model_config,
                seed=0,
                config=exp_config,
            )
            
            self._reset_monitor()
            start = time.time()
            experiment.run()
            duration = time.time() - start
            
            sps = (n_envs * 100 * 5) / duration
            self.results["rl_stress"]["sps"] = sps
            self.results["rl_stress"]["gpu_util"] = self.peak_util
            self.results["rl_stress"]["gpu_mem"] = self.peak_mem
            self._save_results()
            print(f"   RL Throughput: {sps:.0f} SPS | Max Util: {self.peak_util}%")

        except Exception as e:
            print(f"   RL Test Failed: {e}")
            self.results["rl_stress"]["error"] = str(e)
            self._save_results()

    def run_llm_stress(self):
        print("\n🔹 Running LLM/Transformer Stress (Compute & VRAM)...")
        
        # Transformer Loop First (to avoid OOM from the fill test)
        try:
            seq_len = 1024
            hidden_dim = 4096 
            batch_size = 8 # Safer
            
            layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=32, batch_first=True, device=self.device)
            model = torch.nn.Sequential(*[layer for _ in range(8)]).to(self.device)
            optimizer = torch.optim.Adam(model.parameters())
            inputs = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            
            self._reset_monitor()
            for _ in range(5):
                out = model(inputs)
                out.sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            
            start = time.time()
            iters = 20
            for _ in range(iters):
                out = model(inputs)
                out.sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            duration = time.time() - start
            
            tps = (batch_size * seq_len * iters) / duration
            self.results["llm_stress"]["tokens_per_sec"] = tps
            self.results["llm_stress"]["gpu_util"] = self.peak_util
            self.results["llm_stress"]["gpu_mem"] = self.peak_mem
            self._save_results()
            print(f"   Transformer Speed: {tps:.0f} tokens/s | Peak Util: {self.peak_util}%")
        except Exception as e:
            print(f"   Transformer Test Failed: {e}")
            self.results["llm_stress"]["error"] = str(e)
            self._save_results()

        # Then VRAM Fill
        allocated = []
        try:
            torch.cuda.empty_cache()
            # Get current free memory roughly
            r = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], encoding="utf-8")
            free_mb = int(r.strip())
            # Try to fill 90% of it
            target_gb = int(free_mb * 0.9 / 1024)
            print(f"   Attempting to allocate {target_gb}GB VRAM...")
            for _ in range(target_gb):
                allocated.append(torch.empty(256 * 1024 * 1024, dtype=torch.float32, device=self.device))
        except Exception as e:
            print(f"   Allocation reached limit early: {e}")
        
        self.results["llm_stress"]["vram_fill_test"] = f"{len(allocated)} GB"
        self._save_results()
        print(f"   VRAM Successfully Filled: {len(allocated)} GB")
        allocated = None
        torch.cuda.empty_cache()

    def _reset_monitor(self):
        self.peak_mem = 0
        self.peak_util = 0

    def run(self):
        try:
            mp.set_start_method('spawn', force=True)
        except:
            pass

        # Start monitoring
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.run_low_level()
        self._save_results()
        self.run_rl_stress()
        self.run_llm_stress()
        
        self.stop_monitoring = True
        print(f"\n✅ Full Hardware Benchmark Complete. Saved to {self.output_file}")

if __name__ == "__main__":
    bench = GPUStressBenchmark()
    bench.run()
