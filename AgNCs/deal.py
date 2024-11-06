import os
import pandas as pd

# 设定文件夹路径和输出文件路径
folder_path = 'AgNCs'  # 替换为存储多个 Excel 文件的文件夹路径
output_file_path = 'combined_data.xlsx'

# 用于存储所有 Excel 文件中提取的数据
combined_data = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取当前 Excel 文件，确保不将任何列作为索引
        df = pd.read_excel(file_path, sheet_name='Result sheet', skiprows=53, index_col=None)
        print(df)
        # 找到包含 'endtime' 的行，并记录其索引
        end_row = df[df.apply(lambda row: row.astype(str).str.contains('End Time', case=False, na=False).any(), axis=1)].index
        # 如果存在 'endtime'，则仅保留从第54行到 'endtime' 行之间的数据
        print(end_row)
        if not end_row.empty:
            df = df.iloc[:end_row[0]]
        print(df)
        # 清理数据，去除完全空的列和行
        df_cleaned = df.dropna(how='all', axis=1)  # 去掉完全空的列
        df_cleaned = df_cleaned.dropna(how='all')   # 去掉完全空的行
        
        # 将清理后的数据添加到总列表中
        combined_data.append(df_cleaned)


print(combined_data)
# 将所有文件的数据整合到一个 DataFrame 中
if combined_data:
    combined_df = pd.concat(combined_data, ignore_index=True)


    print(combined_df)
    # 将整合后的数据保存到一个新的 Excel 文件中
    combined_df.to_excel(output_file_path, index=False)

    print(f"所有文件的数据已成功提取并保存到 {output_file_path}")
else:
    print("未找到任何有效数据。")

combined_data = []
df = pd.read_excel("./base.xlsx", sheet_name='Result sheet', skiprows=53, index_col=None)
end_row = df[df.apply(lambda row: row.astype(str).str.contains('End Time', case=False, na=False).any(), axis=1)].index
        # 如果存在 'endtime'，则仅保留从第54行到 'endtime' 行之间的数据
print(end_row)
if not end_row.empty:
    df = df.iloc[:end_row[0]]
print(df)
# 清理数据，去除完全空的列和行
df_cleaned = df.dropna(how='all', axis=1)  # 去掉完全空的列
df_cleaned = df_cleaned.dropna(how='all')   # 去掉完全空的行
        
# 将清理后的数据添加到总列表中
combined_data.append(df_cleaned)

if combined_data:
    combined_df = pd.concat(combined_data, ignore_index=True)


    print(combined_df)
    # 将整合后的数据保存到一个新的 Excel 文件中
    combined_df.to_excel("./bias.xlsx", index=False)

    print(f"所有文件的数据已成功提取并保存到 bias.xlsx")