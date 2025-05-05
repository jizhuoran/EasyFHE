import re

equal_count = 0
unequal_indices = []
de_en_index = []
j_11_count = 0

with open('aespa best-89.9.txt', 'r') as file:
    prev_is_for_image = False
    for line in file:
        stripped_line = line.strip()

        if prev_is_for_image:
            # 解析数据行
            data_match = re.fullmatch(
                r'ground truth:\s*(\d+)\s+prediction:\s*(\d+)\s+index:\s*(\d+)',
                stripped_line
            )

            if data_match:
                i = int(data_match.group(1))
                j = int(data_match.group(2))
                index = data_match.group(3)

                if i == j:
                    equal_count += 1
                else:
                    unequal_indices.append(index)

                if j == 11:
                    j_11_count += 1
                    de_en_index.append(index)

            prev_is_for_image = False
        else:
            # 检查是否为For image行
            if re.fullmatch(r'For image\s+\d+', stripped_line):
                prev_is_for_image = True

# 输出结果
print(f"Equal count: {equal_count}")
print("Unequal indices:")
print(', '.join(unequal_indices))
print(f"Number of predictions where j=11: {j_11_count}")
print("Unequal indices:")
print(', '.join(de_en_index))
