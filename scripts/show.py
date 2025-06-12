import torch

if __name__ == "__main__":
    pt_path = "experiments/result.tp1.pt"
    logits_list = torch.load(pt_path)
    print(len(logits_list))

    for idx, logits in enumerate(logits_list):
        print(
            f"{idx}: {logits['timestamp']}, {str(logits['function_name']).split('.')[-1]}, "
            f"{logits['topk_weights'].shape}, {logits['topk_ids'].shape}, {logits['tp_rank']}"
        )

    # tensor([[0.2216, 0.2018, 0.1431, 0.1263, 0.1047, 0.0754, 0.0665, 0.0606]])
    print(logits_list[3]['topk_weights'][0])
    print(logits_list[3]['topk_ids'][0])

    # tensor([[0.2220, 0.2021, 0.1433, 0.1265, 0.1032, 0.0755, 0.0667, 0.0607]])
    # print(logits_list[6]['topk_weights'][0])
    # print(logits_list[6]['topk_ids'][0])
    # print(logits_list[7]['topk_weights'][0])
    # print(logits_list[7]['topk_ids'][0])