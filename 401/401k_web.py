import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from investment_model import InvestmentComparator
from state_tax import StateTaxManager
import os
import pandas as pd
from datetime import datetime
import json
import uuid
from pathlib import Path

def get_contribution_limits(age, annual_income):
    """Calculate all contribution limits based on 2025 rules"""
    # Employee contribution limits
    base_limit = 23500  # 2025 base limit
    if age >= 60 and age <= 63:
        catch_up = 11250
        total_limit = 81250
    elif age >= 50:
        catch_up = 7500
        total_limit = 77500
    else:
        catch_up = 0
        total_limit = 70000
    
    employee_limit = base_limit + catch_up
    
    # Calculate percentage limits for employee contribution
    max_employee_pct = min(100.0, (employee_limit / annual_income) * 100)
    
    return {
        'employee_limit': employee_limit,
        'total_limit': total_limit,
        'max_employee_pct': max_employee_pct
    }

def calculate_match_limit(total_limit, employee_contribution, annual_income, employer_match_pct):
    """Calculate maximum matchable percentage based on remaining room"""
    # Calculate remaining room for employer contributions
    employee_dollar_contribution = annual_income * (employee_contribution/100)
    remaining_contribution_room = total_limit - employee_dollar_contribution
    
    # Convert to matchable percentage (considering employer match rate)
    max_match_limit = (remaining_contribution_room / annual_income) * 100
    max_match_limit = max_match_limit * (100 / employer_match_pct) if employer_match_pct > 0 else 0
    
    return max_match_limit

def get_match_help_text(current_age, annual_income, employer_match, max_match, total_limit):
    """Generate help text for match limit input"""
    return (
        f"The maximum percentage of your salary that can be matched. Example: If they 'match up to 6% of your salary', enter 6. \n\n"
        f"Note: Due to 2025 IRS limits: \n"
        f"• Total contributions (employee + employer) cannot exceed: \n"
        f"  - Age < 50: $70,000 \n"
        f"  - Age 50-59: $77,500 \n"
        f"  - Age 60-63: $81,250 \n\n"
        f"Your Situation: \n"
        f"Age: {current_age} \n"
        f"Income: ${annual_income:,.0f} \n"
        f"Employer Match: {employer_match}% \n"
        f"Maximum matchable percentage: {max_match:.1f}% (based on IRS limits)"
    )

def save_user_inputs(inputs_dict):
    """Save user inputs to a JSON file with timestamp"""
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(data_dir, f'user_inputs_{timestamp}.json')
        
        # Add timestamp to inputs
        inputs_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(inputs_dict, f, indent=4)
        
        return True
    except Exception as e:
        st.error(f"Error saving inputs: {str(e)}")
        return False

def get_or_create_user_id():
    """Get existing user ID from browser storage or create new one"""
    # Check if user_id exists in query parameters (for returning users)
    if 'user_id' in st.query_params:
        user_id = st.query_params['user_id']
        st.session_state.user_id = user_id
        return user_id
    
    # If not in query params, check session state
    if 'user_id' not in st.session_state:
        # Generate new user ID
        new_user_id = str(uuid.uuid4())
        st.session_state.user_id = new_user_id
        # Add user_id to URL for future visits
        st.query_params['user_id'] = new_user_id
    
    return st.session_state.user_id

def log_user_inputs_to_csv(inputs_dict, user_id):
    """Log all user inputs to a single CSV file with user ID"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # Use a single CSV file for all inputs
        csv_path = data_dir / 'all_user_inputs.csv'
        
        # Add user_id to flat_data
        flat_data = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'current_age': inputs_dict['Basic Parameters']['Current Age'],
            'annual_income': inputs_dict['Basic Parameters']['Annual Income'],
            'state': inputs_dict['Basic Parameters']['State'],
            'retirement_age': inputs_dict['Basic Parameters']['Retirement Age'],
            'roth_contribution': inputs_dict['Basic Parameters']['Roth Contribution'].rstrip('%'),
            'employer_match': inputs_dict['Basic Parameters']['Employer Match'].rstrip('%'),
            'match_limit': inputs_dict['Basic Parameters']['Match Limit'].rstrip('%'),
            'inflation_rate': inputs_dict['Market Assumptions']['Inflation Rate'].rstrip('%'),
            'salary_growth': inputs_dict['Market Assumptions']['Salary Growth'].rstrip('%'),
            '401k_return': inputs_dict['Market Assumptions']['401k Return'].rstrip('%'),
            'active_return': inputs_dict['Market Assumptions']['Active Return'].rstrip('%'),
            'passive_return': inputs_dict['Market Assumptions']['Passive Return'].rstrip('%'),
            'roth_value': inputs_dict['Results']['Roth 401k Value'].lstrip('$').replace(',', ''),
            'self_managed_value': inputs_dict['Results']['Self-Managed Value'].lstrip('$').replace(',', ''),
            'difference': inputs_dict['Results']['Difference'].rstrip('%')
        }
        
        df_new = pd.DataFrame([flat_data])
        
        try:
            df_existing = pd.read_csv(csv_path)
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_updated = df_new
        
        df_updated.to_csv(csv_path, index=False)
        
    except Exception as e:
        print(f"Error logging inputs: {str(e)}")

def show_user_history(user_id):
    """Display previous calculations for the current user"""
    try:
        data_dir = Path(__file__).parent / 'data'
        csv_path = data_dir / 'all_user_inputs.csv'
        
        if not csv_path.exists():
            return
        
        df = pd.read_csv(csv_path)
        user_history = df[df['user_id'] == user_id].sort_values('timestamp', ascending=False)
        
        if len(user_history) == 0:
            return
        
        for i, row in user_history.iterrows():
            with st.expander(f"Calculation {len(user_history)-i}: {row['timestamp']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Basic Parameters**")
                    st.write(f"Age: {row['current_age']}")
                    st.write(f"Income: ${float(row['annual_income']):,.0f}")
                    st.write(f"State: {row['state']}")
                    st.write(f"Retirement Age: {row['retirement_age']}")
                    st.write(f"Roth Contribution: {row['roth_contribution']}%")
                    st.write(f"Employer Match: {row['employer_match']}%")
                    st.write(f"Match Limit: {row['match_limit']}%")
                
                with col2:
                    st.markdown("**Market Assumptions**")
                    st.write(f"Inflation Rate: {row['inflation_rate']}%")
                    st.write(f"Salary Growth: {row['salary_growth']}%")
                    st.write(f"401k Return: {row['401k_return']}%")
                    st.write(f"Active Return: {row['active_return']}%")
                    st.write(f"Passive Return: {row['passive_return']}%")
                
                with col3:
                    st.markdown("**Results**")
                    st.metric("Roth 401k", f"${float(row['roth_value']):,.2f}")
                    st.metric("Self-Managed", f"${float(row['self_managed_value']):,.2f}")
                    st.metric("Difference", f"{row['difference']}")
        
        if len(user_history) > 0:
            if st.button("Clear History", key="clear_history_button"):
                df_updated = df[df['user_id'] != user_id]
                df_updated.to_csv(csv_path, index=False)
                st.success("Your calculation history has been cleared!")
                st.rerun()
            
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

def has_previous_calculations(user_id):
    """Check if user has any previous calculations"""
    try:
        data_dir = Path(__file__).parent / 'data'
        csv_path = data_dir / 'all_user_inputs.csv'
        
        if not csv_path.exists():
            return False
        
        df = pd.read_csv(csv_path)
        user_history = df[df['user_id'] == user_id]
        
        return len(user_history) > 0
    except Exception:
        return False

def main():
    # Get or create persistent user ID
    user_id = get_or_create_user_id()
    
    # Initialize session state for calculation results
    if 'roth_value' not in st.session_state:
        st.session_state.roth_value = None
        st.session_state.roth_annual = None
        st.session_state.roth_prob = None
        st.session_state.self_value = None
        st.session_state.self_annual = None
        st.session_state.self_prob = None
        st.session_state.difference_value = None

    # Initialize managers
    state_manager = StateTaxManager()
    
    st.markdown("<h1 style='text-align: center;'>Roth 401(k) vs. Self-Managed Investing: Which Is Right for You?</h1>", unsafe_allow_html=True)
    # Update the introduction section with a table
    st.markdown("""
    <style>
        h1 {
        margin-bottom: 1em;  /* Reduce space after title */
        }
        p {
        margin-bottom: 1em;  /* Reduce space after paragraph */
        }
        .comparison-header {
            text-align: center;
            color: #0f4c81;
            padding: 0;
            margin: 0;
            line-height: 1;
        }
        .feature-table {
            margin: 0;
            padding: 0;
        }
        .feature-header {
            background-color: #f0f2f6;
            font-weight: bold;
        }
        .feature-row:nth-child(even) {
            background-color: #f8f9fa;
        }
        .stMarkdown {
            margin-bottom: 0em;
            padding: 0;
            line-height: 1.2;
        }
        div[data-testid="stTable"] {
            margin: 0;
            padding: 0;
        }
        /* Remove extra space between elements */
        div.element-container {
        margin-bottom: 0;
        padding-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 0.1em;'>This calculator helps you compare two investment strategies, both using the same after-tax contribution amount:</p>", unsafe_allow_html=True)
    st.markdown("<h3 class='comparison-header'>Investment Strategy Comparison</h3>", unsafe_allow_html=True)

    # Create comparison table with improved structure
    comparison_data = {
        "Feature": [
            "💰 Contributions",
            "📈 Growth & Withdrawals",
            "🤝 Employer Benefits",
            "🎯 Investment Options",
            "🎮 Investment Control",
            "💵 Liquidity",
            "📊 Annual Limits (2025)",
            "⚡ Early Withdrawal"
        ],
        "Roth 401(k)": [
            "After-tax dollars",
            "Tax-free",
            "✅ Employer match available",
            "Limited (plan-specific funds)",
            "Limited to plan options",
            "Limited until retirement",
            "$23,500 + catch-up (varies by age)",
            "10% penalty + restrictions"
        ],
        "Self-Managed Portfolio": [
            "After-tax dollars",
            "Subject to capital gains tax",
            "❌ No employer match",
            "Unlimited (stocks, ETFs, etc.)",
            "Full control",
            "Fully liquid",
            "No contribution limits",
            "Available anytime"
        ]
    }

    df = pd.DataFrame(comparison_data)
    st.markdown('<div class="feature-table">', unsafe_allow_html=True)
    st.table(df.set_index('Feature').style
            .set_properties(**{
                'background-color': '#f0f2f6',
                'border': '1px solid #e1e4e8',
                'padding': '0px',
                'text-align': 'left'
            })
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0f4c81'), 
                                           ('color', 'white'),
                                           ('font-weight', 'bold'),
                                           ('padding', '5px')]},
                {'selector': 'td', 'props': [('padding', '5px')]}
            ]))
    st.markdown('</div>', unsafe_allow_html=True)

    # Update the help text styling to be closer to the table
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 0em; border-radius: 5px; margin-top: 0em;'>
        <p style='margin: 0;'><em>💡 Hover over ⓘ icons next to parameters for detailed explanations.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for main interface, history, and methodology
    tab_main, tab_history, tab_methodology = st.tabs(["Investment Calculator", "Calculation History", "Model Methodology"])

    with tab_main:
        # Essential Parameters Section
        st.subheader("Essential Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_age = st.number_input(
                "Current Age", 
                value=22, 
                min_value=18, 
                max_value=65,
                help="Your current age"
            )
            annual_income = st.number_input(
                "Pre-tax Annual Income ($)", 
                value=100000, 
                min_value=0,
                help="Your current pre-tax annual income"
            )
            selected_state = st.selectbox(
                "Select Your State",
                options=state_manager.get_state_list(),
                help="Your state of residence for tax calculations. Different states have different tax treatments for investments."
            )
        
        # Calculate Roth limits
        def get_roth_limit(age):
            base_limit = 23500  # 2025 base limit
            if age >= 60 and age <= 63:
                catch_up = 11250
            elif age >= 50:
                catch_up = 7500
            else:
                catch_up = 0
            return base_limit + catch_up

        roth_limit = get_roth_limit(current_age)
        max_percent = min(100.0, (roth_limit / annual_income) * 100)
        suggested_value = min(10.0, max_percent/2)
        
        with col2:
            retirement_age = st.number_input(
                "Retirement Age", 
                value=65, 
                min_value=current_age + 1,
                help="Your planned retirement age"
            )

            limits = get_contribution_limits(current_age, annual_income)
            roth_contribution_pct = st.slider(
                "Roth 401k Contribution (%)", 
                0.0, 
                limits['max_employee_pct'], 
                suggested_value,
                help=f"Your maximum contribution: ${limits['employee_limit']:,.0f}/year ({limits['max_employee_pct']:.1f}% of income based on 2025 rules)"
            )
            inflation_rate = st.number_input(
                "Inflation Rate (%)",
                value=3.0,
                min_value=0.0,
                max_value=10.0,
                help="Expected annual inflation rate. Historical average is around 3%. This affects the real (inflation-adjusted) value of your investments."
            )
        
        # Calculate maximum employer match based on 2025 rules
        max_match_percent = 100.0  # Maximum match rate is 100% (dollar-for-dollar)
        max_match_limit = 15.0     # Maximum matchable percentage of salary (2025 rule)

        with col3:
            employer_match = st.number_input(
                "Employer Match Percentage", 
                value=50.0,
                min_value=0.0,
                max_value=100.0,
                help="How much your employer matches per dollar you contribute. Example: If they give 50¢ for each dollar you contribute, enter 50. If they match each dollar one-for-one, enter 100."
            )
            
            max_match = calculate_match_limit(
                limits['total_limit'], 
                roth_contribution_pct, 
                annual_income, 
                employer_match
            )
            
            match_limit = st.number_input(
                "Maximum Matchable Salary %", 
                value=min(6.0, max_match),
                min_value=0.0,
                max_value=max_match,
                help=get_match_help_text(
                    current_age, 
                    annual_income, 
                    employer_match, 
                    max_match,
                    limits['total_limit']
                )
            )

            if match_limit > max_match:
                st.warning(f"⚠️ Match limit exceeds 2025 maximum of {max_match}%")

            salary_growth = st.number_input(
                "Annual Salary Growth (%)",
                value=4.0,
                min_value=0.0,
                max_value=15.0,
                help="Expected annual increase in your salary. Typically ranges from 2-5% for cost of living adjustments, potentially higher for career growth."
            )

        # Advanced Parameters (collapsible)
        with st.expander("Adjust Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Investment Returns")
                r_401k = st.number_input(
                    "401k Return Rate (%)", 
                    value=10.0,
                    help="Expected annual return rate for your 401k investments. Default is based on historical S&P 500 returns."
                )
                active_self_managed = st.number_input(
                    "Self-Managed Active Return Rate (%)", 
                    value=12.0,
                    help="Expected return rate for your actively managed investments. Usually higher than passive returns but with more risk and trading costs."
                )
                passive_self_managed = st.number_input(
                    "Self-Managed Passive Return Rate (%)", 
                    value=10.0,
                    help="Expected return rate for your passive investments like index funds. Based on historical market returns."
                )
            
            with col2:
                st.markdown("### Portfolio Structure")
                passive_ratio = st.slider(
                    "Passive Investment Ratio (%)", 
                    0, 100, 80,
                    help="Percentage of self-managed portfolio allocated to passive investments like index funds. Default is 80% passive, 20% active."
                )
                active_trading_freq = st.number_input(
                    "Portfolio Turnover Rate (%)", 
                    value=30.0,
                    help="How often you trade your active investments annually. Higher turnover means more taxable events."
                )
                roth_fee = st.number_input(
                    "Roth 401k Fee (%)", 
                    value=0.2,
                    help="Annual management fee for your Roth 401k investments. Check your plan documents for exact fee structure."
                )

        # Create investment comparator
        comparator = InvestmentComparator()
        
        # Update all parameters including state info
        state_info = state_manager.get_state_info(selected_state)
        comparator.state_tax = state_info['max_rate']
        comparator.state_cg_tax = state_info['max_rate']
        comparator.selected_state = selected_state
        
        # Update other parameters
        comparator.current_age = current_age
        comparator.retirement_age = retirement_age
        comparator.annual_income = annual_income
        comparator.initial_investment = annual_income * (roth_contribution_pct/100)
        comparator.employer_match = employer_match/100  # This is the match rate (50% = 0.5, 100% = 1.0)
        comparator.match_limit = match_limit/100       # This is the salary percentage limit
        comparator.inflation = inflation_rate/100
        comparator.salary_growth = salary_growth/100
        
        # Update advanced parameters if modified
        comparator.r_401k = r_401k/100
        comparator.r_active = active_self_managed/100
        comparator.r_passive = passive_self_managed/100
        comparator.passive_ratio = passive_ratio/100
        comparator.active_ratio = 1 - (passive_ratio/100)
        comparator.active_trading_freq = active_trading_freq/100
        comparator.f_passive = roth_fee/100

        # Calculate button and results display
        if st.button("Calculate Results", key="calculate_results_button"):
            # Run calculations and store in session state
            with st.spinner('*Running investment simulations (this may take a moment)...*'):
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.session_state.roth_value, st.session_state.roth_annual, st.session_state.roth_contributions, st.session_state.roth_prob = comparator.calculate_roth_401k_returns()
                    st.session_state.self_value, st.session_state.self_annual, st.session_state.self_contributions, st.session_state.self_prob = comparator.calculate_self_investment_returns()
                progress_placeholder.empty()
            
            # Calculate and store difference value
            difference_pct = ((st.session_state.self_value/st.session_state.roth_value)-1)*100
            st.session_state.difference_value = f"{'+' if difference_pct > 0 else ''}{difference_pct:.1f}%"
            
            # Save inputs to CSV
            inputs_dict = {
                "Basic Parameters": {
                    "Current Age": current_age,
                    "Annual Income": annual_income,
                    "State": selected_state,
                    "Retirement Age": retirement_age,
                    "Roth Contribution": f"{roth_contribution_pct}%",
                    "Employer Match": f"{employer_match}%",
                    "Match Limit": f"{match_limit}%"
                },
                "Market Assumptions": {
                    "Inflation Rate": f"{inflation_rate}%",
                    "Salary Growth": f"{salary_growth}%",
                    "401k Return": f"{r_401k}%",
                    "Active Return": f"{active_self_managed}%",
                    "Passive Return": f"{passive_self_managed}%"
                },
                "Results": {
                    "Roth 401k Value": f"${st.session_state.roth_value:,.2f}",
                    "Self-Managed Value": f"${st.session_state.self_value:,.2f}",
                    "Difference": st.session_state.difference_value
                }
            }
            log_user_inputs_to_csv(inputs_dict, user_id)
        
        # Only show results if calculations exist
        if st.session_state.roth_value is not None:
            st.header("Investment Comparison Results")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Roth 401k Final Value", f"${st.session_state.roth_value:,.0f}")
            with col2:
                st.metric("Self-Managed Final Value", f"${st.session_state.self_value:,.0f}")
            with col3:
                st.metric("Difference", st.session_state.difference_value, 
                         help="How much more/less the self-managed strategy returns compared to Roth 401k. A positive percentage means self-managed performs better.")

            # Show graph
            st.subheader("Portfolio Value Projections")
            with st.spinner('*Generating visualization...*'):
                fig, ax = plt.subplots(figsize=(10, 6))
                years = range(len(st.session_state.roth_annual))
                
                # Plot mean trajectories and confidence intervals
                ax.plot(years, st.session_state.roth_prob[0], label='Roth 401k (Expected)', color='blue', linewidth=2)
                ax.fill_between(years, st.session_state.roth_prob[1][0], st.session_state.roth_prob[1][1], alpha=0.2, color='blue', label='Roth 401k 90% CI')
                
                ax.plot(years, st.session_state.self_prob[0], label='Self-Managed (Expected)', color='orange', linewidth=2)
                ax.fill_between(years, st.session_state.self_prob[1][0], st.session_state.self_prob[1][1], alpha=0.2, color='orange', label='Self-Managed 90% CI')
                
                ax.set_xlabel('Years')
                ax.set_ylabel('Portfolio Value ($)')
                ax.legend()
                ax.grid(True)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                st.pyplot(fig)

            st.markdown("""
            ### Understanding Your Results

            **Portfolio Projections**
            - Solid lines show the expected growth path for each strategy
            - Shaded areas show where values are likely to fall based on 2,000 market simulations
            - 90% confidence means there's a 90% chance your actual returns will fall within these ranges
            - Wider ranges indicate higher uncertainty due to:
              - Market volatility
              - Active trading exposure
              - Tax implications

            **Key Differences Between Strategies**
            - **Roth 401k**: More predictable due to tax-free growth and lower fees
            - **Self-Managed**: 
              - Higher potential returns but more uncertainty
              - Affected by market timing and tax drag
              - Active self-managed investments add both opportunity and risk
            """)

            # Add age-based contribution info in results section
            st.markdown("""
            ### Your Contribution Limits
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Employee Contribution Limit**
                - Base Limit: $23,500
                - Age-Based Addition: ${limits['employee_limit'] - 23500:,.0f}
                - Your Total Limit: ${limits['employee_limit']:,.0f}
                """)
            with col2:
                st.markdown(f"""
                **Combined Contribution Limit**
                - Maximum Total: ${limits['total_limit']:,.0f}
                - Includes: Employee + Employer Contributions
                - Based on: Age {current_age}
                """)

            # Add warning for high incomes where limits affect percentages
            if max_match < 6.0:  # Standard match becoming limited
                st.info(f"""
                ℹ️ **Note on Contribution Limits**
                Due to your income level (${annual_income:,.0f}), the maximum matchable percentage is limited to {max_match:.1f}% 
                to stay within 2025 IRS limits of ${limits['total_limit']:,.0f} total contributions.
                """)

    with tab_history:
        st.subheader("Compare Calculations")
        if has_previous_calculations(user_id):
            show_user_history(user_id)
        else:
            st.info("Make your first calculation to see your history here!")

    with tab_methodology:
        st.markdown("""
        ### Model Methodology and Assumptions
        
        **Market Modeling**
        - Uses historical S&P 500 returns as baseline (10% average)
        - Incorporates market regimes:
          - Bull Markets: +15% return, 12% volatility
          - Bear Markets: -10% return, 25% volatility
          - Market Crashes: -30% return, 40% volatility
        
        **Investment Assumptions**
        - Monthly compounding of returns
        - Reinvestment of dividends
        - Annual rebalancing of portfolio
        - Inflation adjustment of 3% annually
        
        **Tax Treatment**
        - Federal and state tax rates
        - Long-term vs short-term capital gains
        - Dividend taxation
        - Trading frequency impact
        
        **Model Limitations**
        - Cannot predict future market conditions
        - Assumes consistent contribution patterns
        - Does not account for behavioral factors
        - Tax laws may change
        """)

if __name__ == "__main__":
    main() 