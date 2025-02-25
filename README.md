# Roth 401k vs Self-Managed Investment Comparison

An interactive tool to compare long-term returns between Roth 401k and self-managed investment strategies.
Click here to access the website: https://roth401-or-self-invest.streamlit.app/?user_id=9be41e85-1720-43d4-bae1-ce1d3522fc24 

## Features
- Real-time comparison of investment strategies
- Interactive parameter adjustment
- Tax-aware calculations
- Inflation-adjusted results
- Visual growth comparison

## Installation
```bash
git clone https://github.com/Kohaningithub/Roth-401k-vs-Self-Managed-Investment-Comparison.git
cd Roth-401k-vs-Self-Managed-Investment-Comparison
pip install -r requirements.txt
```

## Usage
```bash
streamlit run 401k_web.py
```

## Model Assumptions
- Returns based on historical S&P 500 average (10% nominal)
- 3% default inflation rate
- Tax rates based on 2024 US tax brackets
- Considers both passive and active investment strategies

## Limitations
- Assumes consistent returns without market volatility
- Does not consider employer matching contributions
- Tax brackets assumed constant over time
- Does not model early withdrawal scenarios

## License
MIT License

## Author
Kohan Chen
