import os
import xarray as xr
import xesmf as xe


def quick_regrid(
    ds_in, ds_out, var_name=None, method="bilinear", periodic=False, reuse_weights=True
):
    """
    xESMF 網格轉換（Regridding）萬用 Helper Function

    參數:
    ----------
    ds_in : xarray.Dataset 或 xarray.DataArray
        輸入的原始數據。必須包含 'lat' 和 'lon' 維度/座標。
    ds_out : xarray.Dataset 或 xarray.DataArray
        目標網格模板。必須包含 'lat' 和 'lon' 維度/座標。
    var_name : str, optional
        若 ds_in 是 Dataset，可指定只轉換特定變數。若為 None 則轉換整份 Dataset。
    method : str, default 'bilinear'
        內插方法。可選：
        - 'bilinear' (雙線性，適合一般氣溫、風速等連續變數)
        - 'nearest_s2d' (最近鄰，適合土地利用類型、掩膜等離散變數)
        - 'conservative' (保守型，適合降雨量、通量等需要總量守恆的變數，網格需有邊界資訊)
    periodic : bool, default False
        是否為全球週期性網格（經度 0~360 或 -180~180 連接處）。
    reuse_weights : bool, default True
        是否重複使用計算好的權重檔案（可大幅加速第二次以後的轉換）。

    返回:
    -------
    xarray.Dataset 或 xarray.DataArray
        轉換到目標網格後的數據。
    """
    # 1. 決定要處理的數據是整個 Dataset 還是特定變數
    if isinstance(ds_in, xr.Dataset) and var_name is not None:
        data_to_regrid = ds_in[var_name]
    else:
        data_to_regrid = ds_in

    # 2. 定義權重檔案名稱（避免多個腳本衝突）
    filename = f"regrid_weights_{method}.nc"

    # 3. 建立或讀取 Regridder 轉換器
    # xESMF 會自動尋找輸入與目標網格中的 'lat' 和 'lon'
    regridder = xe.Regridder(
        ds_in,
        ds_out,
        method=method,
        periodic=periodic,
        filename=filename,
        reuse_weights=reuse_weights,
    )

    # 4. 執行網格轉換
    ds_regridded = regridder(data_to_regrid)

    # 5. 清理產生的權重暫存檔（如果不想留著可以取消註解，但建議留著加速下次執行）
    # if os.path.exists(filename):
    #     os.remove(filename)

    # 6. 釋放記憶體記憶
    regridder.clean_weight_file()

    print(f"✨ 網格轉換完成！使用方法: {method}")
    return ds_regridded
