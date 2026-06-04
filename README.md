
## 启动

```powershell
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY="你的DeepSeek Key"
python server.py
```

打开：

```text
http://127.0.0.1:8501
```

管理员登录：

- 账号：`13246429006`
- 默认密码：`102906`

## 可选环境变量

```powershell
$env:DEEPSEEK_API_KEY="DeepSeek Key"
$env:TUSHARE_TOKEN="Tushare Token"
$env:MAX_OPTIMIZATION_ATTEMPTS="500"
$env:LOOKBACK_DAYS="1825"
$env:HISTORY_MAX_WORKERS="8"
$env:ADMIN_USER="自定义账号"
$env:ADMIN_PASSWORD_SHA256="自定义密码SHA256"
```

## 输出文件

- `data/outputs/latest_predictions.csv`：最新选股结果。
- `data/outputs/latest_validation.csv`：逐日回测验证明细，含 `success_1d/success_5d/success_20d`。
- `data/outputs/latest_optimization_log.csv`：每轮模型优化日志。
- `data/outputs/latest_feature_history.csv`：历史特征样本。

## 页面展示

主页面只展示关键结果字段：

- 股票
- 股票代码
- 预计 1 日涨幅
- 预计 5 日涨幅
- 预计 1 个月涨幅
- 风险率
- 依据
