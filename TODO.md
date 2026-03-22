Input:
Gemma3-4b

Output:
Kinect-Gemma3-4b

Dataset: 
Kinetics-700, ...

Process:
1. Data Preparation
    - Extract frames from video
    - Design description for them with gemma format
2. Model Setup
    - gemma3-4b
    - freeze text layer, unfreeze vision layer
3. SFT
4. Evaluation Benchmark
    - Input: VLM + video
    - Output: 
        - LLM Judger -> score from 0 to 10
            - Based on the accuracy of multiple choice questions
        - 平均每次查詢耗時 (Average Query Latency)
        - 最快 / 最慢耗時 (Min / Max Latency)
        - 🌟 等效 FPS (Frames Per Second)
        - 🌟 等效即時延遲 (Equivalent Real-time Latency)
        - 首字延遲 (Time To First Token, TTFT)
        - 🌟 單字生成平均耗時 (Time Per Output Token, TPOT)
        - 視覺編碼單獨延遲 (Vision Encoding Latency)
        - 🌟 吞吐量 (Throughput: Tokens/sec)
        - 🌟 GPU 最高記憶體佔用 (Peak VRAM Usage)
        - 動態記憶體增長率 (KV Cache Growth Rate)
        - 🌟 瞬時硬體利用率與功耗 (Volatile GPU-Util & Power Consumption)