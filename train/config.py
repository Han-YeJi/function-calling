import argparse

def getConfig():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_data_path', type=str, default="../data/train_data.json", help='Data Path')
    parser.add_argument('--vl_data_path', type=str, default="../data/valid_data.json", help='Data Path')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-7B-Instruct", help='Model Path')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA Rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA Alpha')
    parser.add_argument('--lora_target_modules', type=str, nargs='*', default=["q_proj", "v_proj"], help='LoRA Target Modules')
    parser.add_argument('--max_len', type=int, default=12288, help='training max sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=3, help='Epoch')
    # parser.add_argument('--is_completion_only', action='store_true', help='DataCollatorForCompletionOnlyLM')
    args = parser.parse_args()
    
    return args