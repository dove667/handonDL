import torch
import argparse
from transformers import AutoTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
import numpy as np
import os
from tqdm.auto import tqdm
from collections import defaultdict

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="在 SQuAD 问答任务上微调 BERT 模型。")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="运行模式：'train' (训练) 或 'test' (测试)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="总训练轮数 (仅在 train 模式下有效)。")
    parser.add_argument("--epoch", type=int, default=None,
                        help="测试模式下要加载的 checkpoint 对应的 epoch 编号。")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="训练时每个设备的批次大小。")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="评估时每个设备的批次大小。")
    parser.add_argument("--max_seq_length", type=int, default=384,
                        help="分词后输入序列的最大总长度。")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="当文档过长时，分割成较短片段时的步长。")

    args = parser.parse_args()
    
    output_dir = "model/Bert/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 数据加载与探索
    print("正在加载数据集...")
    # 使用 SQuAD v2 数据集进行问答任务
    # SQuAD v2 包含有答案和无答案的问题，更具挑战性
    dataset = load_dataset("squad_v2")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print("训练集大小:", len(train_dataset))
    print("验证集大小:", len(validation_dataset))

    print("\n训练集示例:")
    print(train_dataset[0])

    # 2. 数据预处理与分词
    # 加载 BERT 的分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # SQuAD 训练数据预处理函数
    def prepare_train_features(examples):
        # 对示例进行分词，使用截断和填充，但保留溢出的 token 以处理长文档。
        # 这会导致一个示例（问题+上下文）可能生成多个特征，每个特征的上下文与前一个特征的上下文有部分重叠。
        # 这样可以确保长文档中的答案不会因为截断而丢失。
        pad_on_right = tokenizer.padding_side == "right" # 判断填充方向，BERT 通常是右填充

        # 对问题和上下文进行分词
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"], # 根据填充方向确定先处理问题还是上下文
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first", # 仅截断第二个序列（通常是上下文）
            max_length=args.max_seq_length, # 最大序列长度
            stride=args.doc_stride, # 步长，用于处理长文档时的重叠
            return_overflowing_tokens=True, # 返回溢出的 token
            return_offsets_mapping=True, # 返回 token 到原始文本的偏移映射
            padding="max_length", # 填充到最大长度
        )

        # 由于一个示例可能生成多个特征，我们需要一个映射来知道每个特征对应哪个原始示例。
        # overflow_to_sample_mapping 提供了这个映射。
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # 偏移映射 (offset_mapping) 提供了连接后的 token 序列中每个 token 对应在原始上下文文本中的起始和结束字符位置。
        # 这对于计算答案的起始和结束 token 位置非常有用。
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 为这些特征标注答案的起始和结束位置！
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # 遍历每个生成的特征
        for i, offsets in enumerate(offset_mapping):
            # 对于无法回答的问题，我们将答案位置标记为 [CLS] token 的索引。
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id) # [CLS] token 的索引

            # 获取对应于上下文的序列（根据 token type IDs）。
            # BERT 输入通常是 [CLS] 问题 [SEP] 上下文 [SEP]，token type IDs 用于区分问题和上下文。
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 一个示例可以生成多个跨度（特征），这是包含当前文本跨度的原始示例的索引。
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index] # 获取原始示例的答案信息

            # 如果没有提供答案（SQuAD v2 中的无答案问题），将答案位置设置为 [CLS] token 的索引。
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 答案在原始上下文中的起始/结束字符索引
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # 当前跨度（特征）在文本中的起始和结束 token 索引。
                token_start_index = 0
                # 找到上下文的起始 token 索引
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                # 找到上下文的结束 token 索引
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 检测答案是否超出当前跨度（特征）的范围。
                # 如果答案的起始字符位置小于当前跨度第一个 token 的起始字符位置，
                # 或者答案的结束字符位置大于当前跨度最后一个 token 的结束字符位置，
                # 则认为答案超出范围，将其标记为 [CLS] 索引。
                if start_char < offsets[token_start_index][0] or end_char > offsets[token_end_index][1]:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # 否则，将 token_start_index 和 token_end_index 移动到答案的两个端点。
                    # 注意：我们将 token_start_index 向后移动 1，因为上下文的第一个 token 通常在索引 1
                    # （索引 0 是 [CLS] token）。
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # SQuAD 验证数据预处理函数
    def prepare_validation_features(examples):
        # 预处理逻辑与训练集类似，但不需要标注答案位置，而是保留原始示例 ID 和偏移映射，以便后续进行后处理和评估。
        pad_on_right = tokenizer.padding_side == "right"
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # 由于一个示例可能生成多个特征，我们需要一个映射来知道每个特征对应哪个原始示例。
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # 为了进行评估，我们需要将预测结果转换回答案文本，这需要访问原始示例。
        # 我们将原始示例的 ID 存储在分词后的数据集中。
        tokenized_examples["example_id"] = [examples["id"][sample_mapping[i]] for i in range(len(tokenized_examples["input_ids"]))]

        # 我们还需要存储偏移映射用于评估。
        tokenized_examples["offset_mapping"] = tokenized_examples.pop("offset_mapping")

        return tokenized_examples


    print("\n正在预处理数据集...")
    # 对训练集和验证集应用预处理函数
    tokenized_train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names)
    tokenized_validation_dataset = validation_dataset.map(prepare_validation_features, batched=True, remove_columns=validation_dataset.column_names)

    # 3. DataLoader构建 (Trainer 内部处理)
    # 数据收集器，用于将批次内的输入填充到批次中的最大长度
    data_collator = default_data_collator

    # 4. 模型定义
    print("\n正在加载 BERT 模型...")
    # 如果是测试模式且指定了 epoch，则从 checkpoint 加载模型
    if args.mode == "test" and args.epoch is not None:
        # 根据用户指定的 epoch 构建 checkpoint 路径
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{args.epoch}")
        if not os.path.exists(checkpoint_path):
             print(f"错误：找不到 checkpoint 路径 {checkpoint_path}。请检查路径和 epoch 编号是否正确。")
             return
        model = BertForQuestionAnswering.from_pretrained(checkpoint_path)
        print(f"已从 checkpoint 加载模型：{checkpoint_path}")
    else:
        # 否则加载预训练的 bert-base-uncased 模型
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        print("已加载基础 BERT 模型。")

    # 定义评估指标 (问答任务的评估比较复杂，通常在预测后进行)
    # 我们将使用验证集 loss 作为训练期间监控的指标。

    # 5. 训练与评估
    print("\n正在设置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,           # 输出目录，用于保存 checkpoint 和训练结果
        num_train_epochs=args.epochs,    # 总训练轮数
        per_device_train_batch_size=args.train_batch_size,  # 训练时每个设备的批次大小
        per_device_eval_batch_size=args.eval_batch_size,   # 评估时每个设备的批次大小
        warmup_steps=500,                # 学习率调度器的预热步数
        weight_decay=0.05,               # 权重衰减强度
        logging_dir="runs/Bert",         # 日志存储目录
        logging_steps=500,                # 每隔多少步记录一次日志
        evaluation_strategy="epoch" if args.mode == "train" else "no", # 训练模式下每个 epoch 评估一次
        save_strategy="epoch" if args.mode == "train" else "no",       # 训练模式下每个 epoch 保存一次 checkpoint
        load_best_model_at_end=True if args.mode == "train" else False,# 训练结束时加载验证集表现最佳的模型
        metric_for_best_model="eval_loss", # 使用验证集 loss 作为判断最佳模型的指标
        greater_is_better=False,         # loss 越低越好
        report_to="none"                 # 禁用向外部服务（如 W&B）报告
    )

    print("正在初始化 Trainer...")
    # 对于问答任务，我们通常不在 Trainer 的训练/评估循环中直接计算 EM/F1 等指标。
    # 评估是通过预测起始/结束位置，然后映射回文本，并使用官方的 SQuAD 评估脚本来完成的。
    # 在训练期间，我们将监控验证集 loss。
    trainer = Trainer(
        model=model,                         # 要训练的 🤗 Transformers 模型实例
        args=training_args,                  # 训练参数
        train_dataset=tokenized_train_dataset if args.mode == "train" else None, # 训练数据集
        eval_dataset=tokenized_validation_dataset if args.mode == "train" else None, # 评估数据集
        tokenizer=tokenizer,                 # 分词器，用于数据收集器
        data_collator=data_collator,         # 数据收集器
    )

    if args.mode == "train":
        print("正在开始训练...")
        # Trainer 会根据 save_strategy 和 metric_for_best_model 自动保存 checkpoint。
        # 它会保存到 output_dir/checkpoint-step_number 和 output_dir/best_model 目录下。
        # Trainer 内部已经包含了根据 eval_loss 保存最佳模型的功能。
        trainer.train()
        print("\n训练完成。")
        print(f"Checkpoint 保存在 {output_dir} 目录下。")

    elif args.mode == "test":
        # 测试模式下的自定义交互式问答循环
        if args.epoch is None:
            print("错误：测试模式下必须通过 --epoch 参数指定要加载的 checkpoint 编号。")
            return

        print(f"正在使用 epoch {args.epoch} 的模型进行交互式测试...")

        # 需要原始验证数据集用于获取上下文和进行映射
        validation_features = tokenized_validation_dataset
        validation_examples = validation_dataset

        # 获取预测结果
        print("正在验证集上获取预测结果...")
        # Trainer.predict 返回一个 PredictionOutput 对象，其中包含预测的 logits
        predictions = trainer.predict(validation_features)
        start_logits, end_logits = predictions.predictions

        # 问答预测结果后处理
        # 这是一个简化的后处理版本，官方的 SQuAD 评估脚本更健壮
        def postprocess_qa_predictions(examples, features, start_logits, end_logits, n_best_size=20, max_answer_length=30):
            # 构建一个从 example ID 到其对应 features 索引列表的映射
            example_to_features = defaultdict(list)
            for idx, feature in enumerate(features):
                example_to_features[feature["example_id"]].append(idx)

            # 存储最终预测结果的字典
            all_predictions = {}

            # 遍历所有原始示例
            for example_index, example in enumerate(tqdm(examples)):
                # 获取与当前示例关联的特征索引列表
                feature_indices = example_to_features[example["id"]]

                min_null_score = None # 用于 SQuAD v2，记录无答案预测的得分
                valid_answers = [] # 存储当前示例所有有效答案跨度的信息

                context = example["context"] # 原始上下文文本

                # 遍历与当前示例关联的所有特征，找到最佳答案跨度
                for feature_index in feature_indices:
                    # 获取当前特征的起始和结束 logits
                    start_logit = start_logits[feature_index]
                    end_logit = end_logits[feature_index]
                    # 当前特征的偏移映射
                    offset_mapping = features[feature_index]["offset_mapping"]

                    # 更新最小无答案得分 (仅 SQuAD v2 需要)
                    # 无答案得分是 [CLS] token 的起始和结束 logits 之和
                    cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                    feature_null_score = start_logit[cls_index] + end_logit[cls_index]
                    if min_null_score is None or min_null_score < feature_null_score:
                        min_null_score = feature_null_score

                    # 找到 logits 最高的 n_best_size 个起始和结束位置
                    start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
                    end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()

                    # 遍历所有可能的起始和结束位置组合
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            # 过滤掉无效的答案跨度：
                            # 1. 索引超出偏移映射范围
                            # 2. 偏移映射为 None (通常是特殊 token)
                            if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                            ):
                                continue
                            # 3. 结束索引小于起始索引 (跨度无效)
                            # 4. 答案长度超过最大允许长度
                            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                                continue

                            # 获取答案在原始上下文中的字符起始和结束位置
                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]

                            # 将有效答案跨度及其得分、文本添加到列表中
                            valid_answers.append(
                                {"offsets": (start_char, end_char), "score": start_logit[start_index] + end_logit[end_index], "text": context[start_char: end_char]}
                            )

                # 如果找到了有效答案，选择得分最高的作为当前示例的最佳答案
                if len(valid_answers) > 0:
                    best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
                else:
                    # 如果没有找到有效答案，将最佳答案设置为空文本和得分为 0
                    best_answer = {"text": "", "score": 0.0}

                # 如果无答案预测的得分高于最佳答案的得分，则预测结果为空字符串（表示无答案）
                if min_null_score is not None and min_null_score > best_answer["score"]:
                     all_predictions[example["id"]] = ""
                else:
                    # 否则，将最佳答案的文本作为预测结果
                    all_predictions[example["id"]] = best_answer["text"]

            return all_predictions

        # 获取文本格式的预测结果
        print("正在进行预测结果后处理...")
        # 注意：这里调用 postprocess_qa_predictions 函数，但其结果 all_predictions 并没有被使用。
        # 这部分代码主要是为了演示如何进行后处理，实际交互测试使用的是下面的代码。
        # 如果需要完整的评估指标 (EM/F1)，需要使用官方的 SQuAD 评估脚本并传入 all_predictions。
        # predictions_text = postprocess_qa_predictions(validation_examples, validation_features, start_logits, end_logits)


        # 交互式测试循环
        print("\n进入交互式测试模式。输入上下文和问题来测试模型。输入 'quit' 退出。")
        while True:
            context_input = input("\n请输入上下文: ")
            if context_input.lower() == 'quit':
                break
            question_input = input("请输入问题: ")
            if question_input.lower() == 'quit':
                break

            # 对自定义输入进行分词
            inputs = tokenizer(
                question_input,
                context_input,
                add_special_tokens=True, # 添加 [CLS] 和 [SEP] 等特殊 token
                return_tensors="pt", # 返回 PyTorch 张量
                max_length=args.max_seq_length, # 最大序列长度
                truncation="only_second", # 仅截断第二个序列（上下文）
                padding="max_length" # 填充到最大长度
            )

            # 将输入数据移动到与模型相同的设备 (CPU 或 GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 获取模型的预测结果 (logits)
            with torch.no_grad(): # 在推理模式下，不计算梯度
                outputs = model(**inputs)

            # 获取起始和结束位置的 logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # 获取最有可能的答案跨度的起始和结束 token 索引
            answer_start_index = torch.argmax(start_logits)
            answer_end_index = torch.argmax(end_logits)

            # 将输入 ID 转换回 token 列表
            input_ids = inputs["input_ids"].squeeze().tolist() # 移除批次维度并转换为列表
            tokens = tokenizer.convert_ids_to_tokens(input_ids) # 将 ID 转换为 token 字符串

            # 获取答案跨度的 token 列表
            answer_tokens = tokens[answer_start_index:answer_end_index + 1]

            # 将答案 token 列表转换回字符串，处理特殊 token 和子词
            answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens), skip_special_tokens=True)

            print(f"\n模型预测答案: {answer}")

if __name__ == "__main__":
    main()
