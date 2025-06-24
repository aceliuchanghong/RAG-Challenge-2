import pandas as pd
import lancedb

# 连接数据库
# python code/count_databse.py
db = lancedb.connect("./lancedb")  # 使用您的数据库路径
table = db.open_table("file_chunks")

# 将整个表加载到 Pandas DataFrame
# 如果表非常大，这可能会消耗很多内存。
print("正在加载数据到 Pandas... (如果数据量大，可能需要一些时间)")
df = table.to_pandas()
print("数据加载完成。")

# 定义唯一标识符列
primary_keys = ["report_sha1", "chunk_id"]

# 查找重复的行
# keep=False 会标记所有重复项，而不仅仅是第二个及以后的
duplicates = df[df.duplicated(subset=primary_keys, keep=False)]

if not duplicates.empty:
    print("\n!!! 发现重复记录 !!!")
    # 按主键排序，以便更容易地看到重复组
    print(duplicates.sort_values(by=primary_keys))
else:
    print("\n恭喜！未在表中发现基于 (report_sha1, chunk_id) 的重复记录。")

# 也可以查看重复项的统计信息
print(f"\n总行数: {len(df)}")
print(f"重复行数: {len(duplicates)}")
print(
    f"唯一 (report_sha1, chunk_id) 组合的数量: {len(df.drop_duplicates(subset=primary_keys))}"
)
