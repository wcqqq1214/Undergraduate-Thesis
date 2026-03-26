import pandas as pd
from pathlib import Path


def load_monitoring_data() -> pd.DataFrame:
    """
    读取项目 data 目录下的 `monitoring data.xlsx`，
    返回完整的 DataFrame，供后续建模和分析使用。
    """
    # 从当前文件位置找到项目根目录，然后定位到 data 目录
    project_root = Path(__file__).parent.parent
    xlsx_path = project_root / "data" / "monitoring data.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {xlsx_path}")

    # 读取第一个工作表的全部数据
    df = pd.read_excel(xlsx_path, sheet_name=0)
    return df
