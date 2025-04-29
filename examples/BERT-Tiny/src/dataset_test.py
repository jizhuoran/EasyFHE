"""
本函数用于从CSV文件中读取一条数据并调用{./FHE-BERT-Tiny "this is a good movie"}来测试
"""
import subprocess
import csv
from dataclasses import dataclass
import argparse

@dataclass
class SplitResult:
    part1: str
    part2: str
    part3: str

def get_input_text(index: int, csv_path: str = "/home/yhh/FYH-BERT-TINY/PNP/FHE-BIRT-TINY/src/data.csv") -> str:
    """从CSV文件中读取指定索引行的文本内容"""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            line = next(row for i, row in enumerate(reader) if i == index)
            return line
        except StopIteration:
            raise IndexError(f"CSV文件只有{index}行，无法读取第{index}行")

def main():
    # 主循环逻辑
    parser = argparse.ArgumentParser(description="数据集获取")
    parser.add_argument("--index", type=int, default=1,help="选择输入index")
    args = parser.parse_args()
    target_index = args.index
    result = get_input_text(index=target_index)
    index = result[0]
    text = result[1]
    print(text)
    label = result[2]
    # print(f"index{index}: |[text]{text}|[label]{label}")
    return text


if __name__ == '__main__':
    main()