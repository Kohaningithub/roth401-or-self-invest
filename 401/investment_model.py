import numpy as np
import matplotlib.pyplot as plt
from state_tax import StateTaxManager
from scipy.stats import norm
import streamlit as st

# Try to import scipy, use fallback if not available
try:
    from scipy import stats
    norm = stats.norm
except ImportError:
    # Fallback implementation of normal distribution
    class NormalDist:
        def rvs(self, loc=0, scale=1, size=None):
            """Generate random normal variates"""
            return np.random.normal(loc=loc, scale=scale, size=size)
        
        def ppf(self, q):
            """Percent point function (inverse of cdf)"""
            if q == 0.05:
                return -1.645  # 5th percentile
            elif q == 0.95:
                return 1.645   # 95th percentile
            return 0  # mean
    norm = NormalDist()

class InvestmentComparator:
    def __init__(self):
        # 基础参数
        self.current_age = 22
        self.retirement_age = 65
        self.investment_years = self.retirement_age - self.current_age
        
        # 投资组合配置
        self.passive_ratio = 0.80  # 80%配置到SPY/QQQ
        self.active_ratio = 0.20   # 20%配置到主动交易
        
        # 交易特征
        self.passive_trading_freq = 0.02  # 被动部分每年2%的换手率
        self.active_trading_freq = 0.30   # 主动部分每年30%的换手率
        self.active_st_ratio = 0.60      # 主动交易中60%是短期
        self.active_lt_ratio = 0.40      # 主动交易中40%是长期

        # 收益率和费用
        self.r_401k = 0.10      # Both use S&P 500 average
        self.r_passive = 0.10   # Both use S&P 500 average
        self.r_active = 0.12   # 主动投资回报率(假设12%)
        self.f_401k = 0.003    # Default 0.3% total fees
        self.f_passive = 0.001  # ETF fee
        self.f_active = 0.002   # Active management fee
        
        # 通货膨胀和增长
        self.inflation = 0.03
        self.salary_growth = 0.04
        
        # 税率
        self.current_tax_rate = 0.24  # 当前所得税率
        self.lt_capital_gains_tax = 0.15  # 长期资本利得税率
        self.st_capital_gains_tax = 0.24  # 短期资本利得税率
        self.dividend_tax = 0.15  # 股息税率
        
        # 股息收益率
        self.passive_dividend_yield = 0.015  # SPY股息率
        self.active_dividend_yield = 0.02   # 主动投资股息率
        
        self.initial_investment = 10000  # 每年投资金额（税后）
        
        self.real_salary_growth = (1 + self.salary_growth) / (1 + self.inflation) - 1
        self.real_return_401k = (1 + self.r_401k) / (1 + self.inflation) - 1
        
        # 添加波动率参数
        self.passive_volatility = 0.15  # SPY历史波动率约15%
        self.active_volatility = 0.25   # 主动投资波动率约25%
        self.correlation = 0.5    # 主动和被动投资的相关性
        self.risk_free_rate = 0.03  # 无风险利率（如美国国债）
        
        self.tax_brackets = [
            (0, 44725, 0.10),      # 2023年税率表
            (44726, 95375, 0.12),
            (95376, 182100, 0.22),
            (182101, 231250, 0.24),
            (231251, 578125, 0.35),
            (578126, float('inf'), 0.37)
        ]
        
        self.annual_income = 100000  # Default annual income
        self.salary_growth = 0.04    # Default salary growth rate
        
        # Add employer match parameters
        self.employer_match = 0.03  # 3% default match
        self.match_limit = 0.06    # Up to 6% of salary
        
        # Add state tax parameters
        self.state_tax = 0.05  # Default 5% state income tax
        self.state_cg_tax = 0.05  # Default 5% state capital gains tax
        
        # Add state tax manager
        self.state_manager = StateTaxManager()
        self.selected_state = "California"  # Default state
        
        # Add market regime parameters
        self.regime_params = {
            'bull': {
                'mean_return': 0.15,  # 15% annual return in bull markets
                'std_dev': 0.12      # 12% volatility
            },
            'bear': {
                'mean_return': -0.10,  # -10% annual return in bear markets
                'std_dev': 0.25       # 25% volatility
            },
            'crash': {
                'mean_return': -0.30,  # -30% annual return during crashes
                'std_dev': 0.40       # 40% volatility
            }
        }
    
    def calculate_investment_years(self):
        """Recalculate investment years when age changes"""
        return self.retirement_age - self.current_age
    
    def calculate_marginal_tax(self, income):
        """计算累进所得税"""
        total_tax = 0
        remaining_income = income
        
        for lower, upper, rate in self.tax_brackets:
            if remaining_income <= 0:
                break
            taxable_amount = min(upper - lower, remaining_income)
            total_tax += taxable_amount * rate
            remaining_income -= taxable_amount
        
        return total_tax / income  # 返回有效税率

    def calculate_roth_401k_returns(self):
        """Calculate both deterministic and probabilistic Roth 401k returns"""
        # Get deterministic results
        det_value, det_annual = self.get_deterministic_results(is_roth=True)
        
        # Get probabilistic results
        prob_value, prob_trajectory, confidence_bounds = self.run_monte_carlo_simulation(is_roth=True)
        
        total_contributions = sum([self.initial_investment * (1 + self.salary_growth)**year 
                                 for year in range(self.calculate_investment_years())])
        
        return det_value, det_annual, total_contributions, (prob_trajectory, confidence_bounds)

    def calculate_self_investment_returns(self):
        """Calculate both deterministic and probabilistic self-managed returns"""
        # Get deterministic results
        det_value, det_annual = self.get_deterministic_results(is_roth=False)
        
        # Get probabilistic results
        prob_value, prob_trajectory, confidence_bounds = self.run_monte_carlo_simulation(is_roth=False)
        
        total_contributions = sum([self.initial_investment * (1 + self.salary_growth)**year 
                                 for year in range(self.calculate_investment_years())])
        
        return det_value, det_annual, total_contributions, (prob_trajectory, confidence_bounds)

    def calculate_true_marginal_tax(self, income, contribution=None):
        """Calculate true marginal tax rate on additional income"""
        if contribution is None:
            contribution = income * 0.01  # Use 1% for marginal calculation
        
        # Calculate federal tax at both income levels
        fed_tax_without = self.calculate_total_tax(income, include_state=False)
        fed_tax_with = self.calculate_total_tax(income + contribution, include_state=False)
        
        # True marginal rate is the difference
        fed_marginal = (fed_tax_with - fed_tax_without) / contribution
        
        # Add state marginal rate
        state_marginal = self.state_manager.get_state_tax_rate(
            self.selected_state,
            income + contribution
        )
        
        return fed_marginal + state_marginal

    def calculate_total_tax(self, income, include_state=True):
        """Calculate total tax including optional state tax"""
        # Federal tax calculation
        total_tax = 0
        remaining_income = income
        
        for lower, upper, rate in self.tax_brackets:
            if remaining_income <= 0:
                break
            taxable_amount = min(upper - lower, remaining_income)
            total_tax += taxable_amount * rate
            remaining_income -= taxable_amount
        
        # Add state tax if requested
        if include_state:
            total_tax += income * self.state_tax
        
        return total_tax

    def calculate_portfolio_return(self):
        """计算投资组合的综合净收益率"""
        passive_return = self.r_passive - self.f_passive
        active_return = self.r_active - self.f_active
        
        portfolio_return = (
            self.passive_ratio * passive_return +
            self.active_ratio * active_return
        )
        return portfolio_return

    def sensitivity_analysis(self):
        """Extended sensitivity analysis with realistic scenarios"""
        plt.figure(figsize=(15, 10))
        
        # 1. Active returns including underperformance
        plt.subplot(2, 2, 1)
        active_returns = [0.08, 0.09, 0.10, 0.11, 0.12]  # 10% is market return
        values_by_return = []
        
        original_active_return = self.r_active
        for r in active_returns:
            self.r_active = r
            value, _, _ = self.calculate_self_investment_returns()
            values_by_return.append(value)
        
        self.r_active = original_active_return
        
        plt.plot(active_returns, values_by_return, marker='o')
        plt.axvline(x=0.10, color='r', linestyle='--', label='Market Return')
        plt.title('Impact of Active Return (Including Underperformance)')
        plt.xlabel('Active Investment Return')
        plt.ylabel('Final Real Value ($)')
        plt.grid(True)
        plt.legend()

        # 2. 主动交易频率的敏感性
        plt.subplot(2, 2, 2)
        active_freqs = [0.1, 0.2, 0.3, 0.4, 0.5]
        values_by_freq = []
        
        original_active_freq = self.active_trading_freq
        for freq in active_freqs:
            self.active_trading_freq = freq
            value, _, _ = self.calculate_self_investment_returns()
            values_by_freq.append(value)
        
        self.active_trading_freq = original_active_freq
        
        plt.plot(active_freqs, values_by_freq, marker='s', color='red')
        plt.title('Impact of Active Trading Frequency')
        plt.xlabel('Active Trading Frequency')
        plt.ylabel('Final Real Value ($)')
        plt.ylim(min(values_by_freq) * 0.95, max(values_by_freq) * 1.05)  # Normalize Y-axis
        plt.grid(True)

        # 3. 主动交易频率的敏感性
        plt.subplot(2, 2, 3)
        active_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        values_by_active_ratio = []
        
        original_active_ratio = self.active_ratio
        original_passive_ratio = self.passive_ratio
        
        for ratio in active_ratios:
            self.active_ratio = ratio
            self.passive_ratio = 1 - ratio
            value, _, _ = self.calculate_self_investment_returns()
            values_by_active_ratio.append(value)
        
        self.active_ratio = original_active_ratio
        self.passive_ratio = original_passive_ratio
        
        plt.plot(active_ratios, values_by_active_ratio, marker='d', color='green')
        plt.title('Impact of Active Investment Allocation')
        plt.xlabel('Active Investment Allocation')
        plt.ylabel('Final Real Value ($)')
        plt.ylim(min(values_by_active_ratio) * 0.95, max(values_by_active_ratio) * 1.05)  # Normalize Y-axis
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def simulate_market_cycles(self, years):
        """Generate market cycle data with Markov Chain transitions"""
        cycles = []
        current_year = 0
        
        # Market regime transition probabilities
        transition_matrix = {
            'bull': {'bull': 0.8, 'bear': 0.15, 'crash': 0.05},
            'bear': {'bull': 0.3, 'bear': 0.6, 'crash': 0.1},
            'crash': {'bull': 0.5, 'bear': 0.4, 'crash': 0.1}
        }
        
        # Start in bull market (most common starting point)
        current_regime = 'bull'
        
        while current_year < years:
            duration = None
            if current_regime == 'bull':
                duration = np.random.randint(24, 60) / 12  # 2-5 years
            elif current_regime == 'bear':
                duration = np.random.randint(12, 24) / 12  # 1-2 years
            else:  # crash
                duration = np.random.randint(6, 12) / 12  # 6-12 months
            
            # Adjust duration if it would exceed simulation period
            if current_year + duration > years:
                duration = years - current_year
            
            cycles.append({
                'regime': current_regime,
                'duration': duration,
                'start_year': current_year,
                'params': self.regime_params[current_regime]
            })
            
            current_year += duration
            
            # Determine next regime
            next_probs = transition_matrix[current_regime]
            current_regime = np.random.choice(
                list(next_probs.keys()),
                p=list(next_probs.values())
            )
        
        return cycles

    def simulate_investment_returns(self, is_roth=False, num_simulations=500, market_cycles=None):
        """Simulate investment returns using provided or new market cycles"""
        investment_years = self.calculate_investment_years()
        all_trajectories = []
        
        print("\nSimulating investment trajectories...")
        
        # Generate market cycles once if not provided
        if market_cycles is None:
            market_cycles = self.simulate_market_cycles(investment_years)
        
        for sim in range(num_simulations):
            total_value = 0
            annual_values = []
            
            for year in range(investment_years):
                # Find applicable market cycle
                cycle = next(c for c in market_cycles if 
                            c['start_year'] <= year < c['start_year'] + c['duration'])
                regime_data = cycle['params']
                
                # Generate monthly returns
                if is_roth:
                    monthly_returns = norm.rvs(
                        loc=regime_data['mean_return']/12,
                        scale=regime_data['std_dev']/np.sqrt(12),
                        size=12
                    )
                else:
                    # Generate correlated returns for passive/active
                    rho = self.correlation
                    cov_matrix = np.array([[1, rho], [rho, 1]])
                    L = np.linalg.cholesky(cov_matrix)
                    
                    uncorrelated = np.random.normal(size=(2, 12))
                    correlated = np.dot(L, uncorrelated)
                    
                    passive_returns = (
                        correlated[0] * regime_data['std_dev']/np.sqrt(12) +
                        regime_data['mean_return']/12
                    )
                    
                    active_premium = (self.r_active - self.r_passive)/12
                    active_returns = (
                        correlated[1] * regime_data['std_dev']*1.5/np.sqrt(12) +
                        regime_data['mean_return']/12 +
                        active_premium
                    )
                
                # Process monthly returns
                for month in range(12):
                    monthly_contribution = (self.initial_investment * 
                                         (1 + self.salary_growth)**year) / 12
                    
                    if is_roth:
                        if total_value > 0:
                            total_value *= (1 + monthly_returns[month] - self.f_401k/12)
                        total_value += monthly_contribution
                        
                        match = self.calculate_employer_match(
                            monthly_contribution,
                            self.annual_income * (1 + self.salary_growth)**year / 12
                        )
                        if match > 0:
                            total_value += match * (1 + monthly_returns[month] - self.f_401k/12)
                    else:
                        if total_value > 0:
                            passive_value = total_value * self.passive_ratio
                            active_value = total_value * self.active_ratio
                            
                            passive_growth = passive_value * (1 + passive_returns[month] - self.f_passive/12)
                            active_growth = active_value * (1 + active_returns[month] - self.f_active/12)
                            
                            tax_impact = (
                                self.calculate_investment_taxes(passive_value, passive_returns[month], False) +
                                self.calculate_investment_taxes(active_value, active_returns[month], True)
                            ) / 12
                            
                            total_value = passive_growth + active_growth - tax_impact
                        
                        total_value += monthly_contribution
                
                # Store annual value
                real_value = total_value / (1 + self.inflation)**(year + 1)
                annual_values.append(real_value)
            
            all_trajectories.append(annual_values)
            if sim % 100 == 0:
                print(f"Completed {sim}/{num_simulations} simulations")
        
        return self.process_simulation_results(all_trajectories)

    def process_simulation_results(self, trajectories):
        """Process and summarize simulation results"""
        trajectories = np.array(trajectories)
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)
        
        ci_lower = np.percentile(trajectories, 5, axis=0)
        ci_upper = np.percentile(trajectories, 95, axis=0)
        
        print("\nSimulation Statistics:")
        print(f"Final Value (mean): ${mean_trajectory[-1]:,.2f}")
        print(f"Standard Deviation: ${std_trajectory[-1]:,.2f}")
        print(f"90% Confidence Interval: ${ci_lower[-1]:,.2f} to ${ci_upper[-1]:,.2f}")
        
        return mean_trajectory[-1], mean_trajectory, trajectories

    def print_comprehensive_analysis(self):
        """Print comprehensive analysis using consistent market cycles"""
        print("\n=== Comprehensive Investment Analysis ===")
        
        # Generate market cycles once
        investment_years = self.calculate_investment_years()
        market_cycles = self.simulate_market_cycles(investment_years)
        
        # Analyze market cycles
        print("\nMarket Cycle Analysis:")
        regime_counts = {'bull': 0, 'bear': 0, 'crash': 0}
        regime_durations = {'bull': [], 'bear': [], 'crash': []}
        
        for cycle in market_cycles:
            regime = cycle['regime']
            duration = cycle['duration']
            regime_counts[regime] += 1
            regime_durations[regime].append(duration)
        
        print("\nMarket Regime Distribution:")
        total_cycles = sum(regime_counts.values())
        for regime, count in regime_counts.items():
            avg_duration = np.mean(regime_durations[regime])
            print(f"{regime.title()} Markets:")
            print(f"  Frequency: {count/total_cycles:.1%}")
            print(f"  Avg Duration: {avg_duration:.1f} years")
        
        # Run investment simulations using the same market cycles
        print("\nRunning Investment Simulations...")
        roth_value, roth_annual, _ = self.simulate_investment_returns(
            is_roth=True, market_cycles=market_cycles)
        self_value, self_annual, _ = self.simulate_investment_returns(
            is_roth=False, market_cycles=market_cycles)
        
        # Print results
        print("\nFinal Portfolio Values (Inflation-Adjusted):")
        print(f"Roth 401k: ${roth_value:,.2f}")
        print(f"Self-Managed: ${self_value:,.2f}")
        print(f"Difference: {((self_value/roth_value)-1)*100:.1f}%")

    def calculate_effective_tax_rate(self, value, gains, trading_freq, is_active=False):
        """计算投资的实际税负"""
        trading_gains = value * trading_freq * gains
        
        if is_active:
            st_tax = trading_gains * 0.6 * self.st_capital_gains_tax
            lt_tax = trading_gains * 0.4 * self.lt_capital_gains_tax
            return st_tax + lt_tax
        else:
            return trading_gains * self.lt_capital_gains_tax

    def projected_capital_gains_tax(self, retirement_income, is_long_term=True):
        """Calculate capital gains tax rate based on income level"""
        if not is_long_term:
            # Short-term gains are taxed as ordinary income
            return self.calculate_true_marginal_tax(retirement_income)
        
        # 2025 Long-term capital gains brackets (single filer)
        cg_brackets = [
            (0, 48350, 0.0),        # 0% bracket
            (48351, 533400, 0.15),  # 15% bracket
            (533401, float('inf'), 0.20)  # 20% bracket
        ]
        
        # Find applicable capital gains rate
        base_rate = 0.0
        for lower, upper, rate in cg_brackets:
            if retirement_income > lower and retirement_income <= upper:
                base_rate = rate
                break
            elif retirement_income > upper:
                base_rate = rate
        
        # Add 3.8% NIIT for high incomes (threshold adjusted for 2025)
        niit_threshold = 200000  # Single filer threshold
        if retirement_income > niit_threshold:
            base_rate += 0.038
        
        # Add state capital gains tax
        state_rate = self.state_manager.get_state_tax_rate(
            self.selected_state,
            retirement_income
        )
        
        return base_rate + state_rate

    def calculate_capital_gains_tax(self, gains, income, is_long_term=True):
        """Calculate capital gains tax using progressive brackets"""
        if not is_long_term:
            # Short-term gains are taxed as ordinary income
            return gains * self.calculate_true_marginal_tax(income + gains)
        
        # 2025 Long-term capital gains brackets (single filer)
        total_tax = 0
        remaining_gains = gains
        total_income = income + gains
        
        cg_brackets = [
            (0, 48350, 0.0),        # 0% bracket
            (48351, 533400, 0.15),  # 15% bracket
            (533401, float('inf'), 0.20)  # 20% bracket
        ]
        
        # Calculate tax for each bracket
        for lower, upper, rate in cg_brackets:
            if total_income > lower:
                # Calculate taxable amount in this bracket
                bracket_income = min(total_income, upper) - lower
                taxable_gains = min(remaining_gains, bracket_income)
                if taxable_gains > 0:
                    total_tax += taxable_gains * rate
                    remaining_gains -= taxable_gains
            
            if remaining_gains <= 0:
                break
        
        # Add NIIT for high incomes
        if total_income > 200000:  # 2025 threshold
            total_tax += gains * 0.038
        
        # Add state tax
        state_rate = self.state_manager.get_state_tax_rate(
            self.selected_state,
            total_income
        )
        total_tax += gains * state_rate
        
        return total_tax

    def calculate_employer_match(self, contribution, current_salary):
        """Calculate employer match based on contribution and limits"""
        # contribution is the employee's contribution for this period
        # current_salary is the salary for this period
        
        # Calculate the contribution as a percentage of salary
        contribution_percent = contribution / current_salary
        
        # If contributing less than match limit, match the full contribution
        if contribution_percent <= self.match_limit:
            return contribution * self.employer_match
        
        # If contributing more than match limit, only match up to the limit
        return (current_salary * self.match_limit) * self.employer_match

    def calculate_investment_taxes(self, value, gains, is_active=False):
        """Calculate investment taxes with proper treatment of gains"""
        total_tax = 0
        
        # Handle dividends
        if is_active:
            dividend_yield = self.active_dividend_yield
        else:
            dividend_yield = self.passive_dividend_yield
        
        dividend_income = value * dividend_yield
        dividend_tax_rate = self.projected_capital_gains_tax(
            self.annual_income + dividend_income,
            is_long_term=True  # Qualified dividends
        )
        total_tax += dividend_income * dividend_tax_rate
        
        # Handle realized gains
        if is_active:
            trading_freq = self.active_trading_freq
            # 60% short-term, 40% long-term for active trading
            realized_gains = value * trading_freq * gains
            st_gains = realized_gains * 0.60
            lt_gains = realized_gains * 0.40
            
            # Calculate tax rates based on income + realized gains
            total_income = self.annual_income + st_gains + lt_gains
            st_tax_rate = self.projected_capital_gains_tax(total_income, is_long_term=False)
            lt_tax_rate = self.projected_capital_gains_tax(total_income, is_long_term=True)
            
            total_tax += (st_gains * st_tax_rate + lt_gains * lt_tax_rate)
        else:
            # Passive investments: mostly long-term gains
            trading_freq = self.passive_trading_freq
            realized_gains = value * trading_freq * gains
            lt_tax_rate = self.projected_capital_gains_tax(
                self.annual_income + realized_gains,
                is_long_term=True
            )
            total_tax += realized_gains * lt_tax_rate
        
        return total_tax

    def run_monte_carlo_simulation(self, is_roth=False, num_runs=4, num_simulations=500):
        """Run Monte Carlo with market regimes and consistent returns"""
        investment_years = self.calculate_investment_years()
        all_results = []
        all_trajectories = []
        
        # Use st.write for real-time updates
        st.write(f"\nRunning {num_runs} sets of {num_simulations} simulations...")
        
        for run in range(num_runs):
            run_trajectories = []
            
            for sim in range(num_simulations):
                # Generate market cycles for entire simulation period
                market_cycles = self.simulate_market_cycles(investment_years)
                total_value = 0
                annual_values = []
                
                for year in range(investment_years):
                    # Find current market regime
                    cycle = next(c for c in market_cycles if 
                               c['start_year'] <= year < c['start_year'] + c['duration'])
                    regime_data = cycle['params']
                    
                    # Generate monthly returns based on regime
                    monthly_market = np.random.normal(
                        loc=regime_data['mean_return']/12,
                        scale=regime_data['std_dev']/np.sqrt(12),
                        size=12
                    )
                    
                    # Generate active returns with higher volatility in each regime
                    if not is_roth:
                        monthly_active = np.random.normal(
                            loc=(regime_data['mean_return'] + (self.r_active - self.r_passive))/12,
                            scale=regime_data['std_dev']*1.5/np.sqrt(12),
                            size=12
                        )
                    
                    year_total_value = total_value
                    contribution = self.initial_investment * (1 + self.salary_growth)**year
                    
                    if is_roth:
                        # Process Roth with regime-based returns
                        for month in range(12):
                            if year_total_value > 0:
                                year_total_value *= (1 + monthly_market[month] - self.f_passive/12)
                            year_total_value += contribution/12
                            
                            match = self.calculate_employer_match(
                                contribution/12,
                                self.annual_income * (1 + self.salary_growth)**year / 12
                            )
                            if match > 0:
                                year_total_value += match * (1 + monthly_market[month] - self.f_passive/12)
                    else:
                        # Process self-managed with regime-based returns
                        for month in range(12):
                            passive_value = year_total_value * self.passive_ratio
                            active_value = year_total_value * self.active_ratio
                            
                            if year_total_value > 0:
                                passive_growth = passive_value * (1 + monthly_market[month] - self.f_passive/12)
                                active_growth = active_value * (1 + monthly_active[month] - self.f_active/12)
                                year_total_value = passive_growth + active_growth
                            
                            year_total_value += contribution/12
                        
                        # Apply annual tax impact
                        if year_total_value > 0:
                            passive_value = year_total_value * self.passive_ratio
                            active_value = year_total_value * self.active_ratio
                            
                            # Calculate realized gains based on regime-appropriate returns
                            passive_gains = passive_value * self.passive_trading_freq * np.mean(monthly_market)
                            active_gains = active_value * self.active_trading_freq * np.mean(monthly_active)
                            
                            tax_impact = (
                                self.calculate_investment_taxes(passive_value, passive_gains/passive_value, False) +
                                self.calculate_investment_taxes(active_value, active_gains/active_value, True)
                            )
                            year_total_value -= tax_impact
                    
                    total_value = year_total_value
                    real_value = total_value / (1 + self.inflation)**(year + 1)
                    annual_values.append(real_value)
                
                run_trajectories.append(annual_values)
            
            # Process run statistics
            run_trajectories = np.array(run_trajectories)
            mean_trajectory = np.mean(run_trajectories, axis=0)
            all_trajectories.append(mean_trajectory)
            all_results.append(mean_trajectory[-1])
            
            # Update progress
            progress = (run * num_simulations + sim) / (num_runs * num_simulations)
            st.progress(progress)
        
        # Calculate final statistics
        final_mean = np.mean(all_results)
        final_std = np.std(all_results)
        ci_lower = np.percentile(all_results, 5)
        ci_upper = np.percentile(all_results, 95)
        
        st.write(f"\nFinal Results:")
        st.write(f"Mean Value: ${final_mean:,.2f}")
        st.write(f"Standard Deviation: ${final_std:,.2f}")
        st.write(f"90% Confidence Interval:")
        st.write(f"• Lower bound: ${ci_lower:,.2f}")
        st.write(f"• Upper bound: ${ci_upper:,.2f}")
        
        mean_trajectory = np.mean(all_trajectories, axis=0)
        ci_lower_trajectory = np.percentile(all_trajectories, 5, axis=0)
        ci_upper_trajectory = np.percentile(all_trajectories, 95, axis=0)
        
        return final_mean, mean_trajectory, (ci_lower_trajectory, ci_upper_trajectory)

    def get_deterministic_results(self, is_roth=False):
        """Calculate deterministic results using expected returns"""
        investment_years = self.calculate_investment_years()
        total_value = 0
        annual_values = []
        
        # Use constant expected returns instead of market cycles
        expected_return = self.r_401k  # Base market return for both strategies
        
        for year in range(investment_years):
            year_total_value = total_value
            contribution = self.initial_investment * (1 + self.salary_growth)**year
            
            if is_roth:
                # Simple growth with fees
                if year_total_value > 0:
                    year_total_value *= (1 + expected_return - self.f_passive)
                year_total_value += contribution
                
                # Add employer match
                match = self.calculate_employer_match(
                    contribution,
                    self.annual_income * (1 + self.salary_growth)**year
                )
                if match > 0:
                    year_total_value += match * (1 + expected_return - self.f_passive)
            else:
                # Split between passive and active
                passive_value = year_total_value * self.passive_ratio
                active_value = year_total_value * self.active_ratio
                
                if year_total_value > 0:
                    # Passive portion (only annual fees)
                    passive_growth = passive_value * (1 + expected_return - self.f_passive)
                    
                    # Active portion (additional return and fees)
                    active_growth = active_value * (1 + expected_return + 
                        (self.r_active - self.r_passive) - self.f_active)
                    
                    # Tax impact from trading
                    passive_tax = self.calculate_investment_taxes(
                        passive_value, 
                        self.passive_trading_freq * expected_return,  # Only tax realized gains
                        False
                    )
                    active_tax = self.calculate_investment_taxes(
                        active_value,
                        self.active_trading_freq * (expected_return + (self.r_active - self.r_passive)),
                        True
                    )
                    
                    year_total_value = passive_growth + active_growth - passive_tax - active_tax
                
                year_total_value += contribution
            
            total_value = year_total_value
            real_value = total_value / (1 + self.inflation)**(year + 1)
            annual_values.append(real_value)
        
        return total_value, annual_values
