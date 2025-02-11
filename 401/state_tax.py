import pandas as pd

class StateTaxManager:
    def __init__(self):
        # Load state tax data
        self.tax_data = pd.read_csv('401/State_Tax_2024.csv')
        
        # Clean up state names (remove footnotes) - use string split instead of regex
        self.tax_data['State'] = self.tax_data['State'].apply(lambda x: x.split(' (')[0])
        
        # Create lookup dictionaries
        self.state_brackets = {}
        self.state_deductions = {}
        
        # Process each state's data
        for _, row in self.tax_data.iterrows():
            state = row['State']
            
            # Skip if already processed (take first entry for states with multiple brackets)
            if state in self.state_brackets:
                continue
                
            # Store standard deduction and exemption
            self.state_deductions[state] = {
                'standard_deduction_single': self._parse_amount(row['Standard Deduction (Single)']),
                'standard_deduction_couple': self._parse_amount(row['Standard Deduction (Couple)']),
                'personal_exemption_single': self._parse_amount(row['Personal Exemption (Single)']),
                'personal_exemption_couple': self._parse_amount(row['Personal Exemption (Couple)'])
            }
            
            # Get all brackets for this state
            state_rows = self.tax_data[self.tax_data['State'] == state]
            brackets = []
            for _, bracket_row in state_rows.iterrows():
                rate = self._parse_rate(bracket_row['Single Filer Rates'])
                income = self._parse_amount(bracket_row['Single Filer Brackets'])
                if rate is not None and income is not None:
                    brackets.append((income, rate))
            
            # Sort brackets by income
            brackets.sort(key=lambda x: x[0])
            self.state_brackets[state] = brackets
    
    def _parse_rate(self, rate_str):
        """Parse rate string to float"""
        try:
            if pd.isna(rate_str) or rate_str == 'none':
                return None
            
            # Handle percentage strings
            if isinstance(rate_str, str):
                # Remove % and convert to float
                rate = rate_str.replace('%', '').strip()
                return float(rate) / 100 if rate else None
            
            # Handle numeric values
            if isinstance(rate_str, (int, float)):
                return float(rate_str) / 100
            
            return None
        except:
            return None
    
    def _parse_amount(self, amount_str):
        """Parse amount string to float"""
        if pd.isna(amount_str) or amount_str == 'n.a.':
            return 0
        
        try:
            # Handle credit format
            if isinstance(amount_str, str) and 'credit' in amount_str:
                # Extract number between $ and space/end
                amount = ''.join(c for c in amount_str if c.isdigit() or c == '.')
                return float(amount) if amount else 0
            
            # Handle regular amounts
            if isinstance(amount_str, str):
                # Remove $ and , and convert to float
                amount = amount_str.replace('$', '').replace(',', '')
                return float(amount) if amount else 0
            
            # Handle numeric values
            if isinstance(amount_str, (int, float)):
                return float(amount_str)
            
            return 0
        except:
            return 0
    
    def get_state_tax_rate(self, state, income):
        """Get marginal tax rate for given state and income"""
        if state not in self.state_brackets:
            return 0.0
            
        brackets = self.state_brackets[state]
        if not brackets:  # No tax brackets (e.g., Texas)
            return 0.0
            
        # Find applicable bracket
        for i, (bracket_income, rate) in enumerate(brackets):
            if income <= bracket_income or i == len(brackets) - 1:
                return rate
        
        return brackets[-1][1]  # Return highest bracket rate
    
    def get_state_info(self, state):
        """Get state tax information"""
        if state not in self.state_brackets:
            return {
                'has_income_tax': False,
                'max_rate': 0.0,
                'num_brackets': 0,
                'standard_deduction': 0,
                'personal_exemption': 0
            }
        
        brackets = self.state_brackets[state]
        deductions = self.state_deductions[state]
        
        return {
            'has_income_tax': bool(brackets),
            'max_rate': max(rate for _, rate in brackets) if brackets else 0.0,
            'num_brackets': len(brackets),
            'standard_deduction': deductions['standard_deduction_single'],
            'personal_exemption': deductions['personal_exemption_single']
        }
    
    def get_state_list(self):
        """Get sorted list of states"""
        return sorted(self.state_brackets.keys()) 