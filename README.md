# A股全市场量化选股平台

本项目是本地 HTML 网站 + FastAPI 后端，用于抓取沪深全市场 A 股数据，训练 1 日、5 日、20 日收益预测模型，并用历史逐日回测验证预测误差。

> 输出仅用于量化研究，不构成投资建议。系统会真实展示验证成功率，不会伪造 95% 通过。

## 核心能力

- 沪深全市场股票池，ST、停牌、新股、数据缺口股票不剔除，只做风险标记。
- 默认抓取最近 5 年历史行情。
- 多源兜底：AKShare/东方财富、腾讯、efinance、Baostock、Tushare 可选 Token。
- 事件数据能抓则抓：新闻、公告、政策新闻、资金流、热点、人气关键词、研报、扩产设厂、订单、中标、回购、减持、处罚等。
- 模型至少优化 100 轮，默认最多 500 轮；达到门槛后才允许提前停止。
- 验证标准：预测收益与真实收益绝对误差 `<= 0.005` 即成功，分别统计 1 日、5 日、20 日。
- DeepSeek 只用于候选股票的新闻/政策/公告依据复核，不直接替代可验证数值模型。

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
