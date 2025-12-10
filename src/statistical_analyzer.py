"""
Модуль для проведения статистического анализа A/B тестов.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import proportion
from statsmodels.stats.power import NormalIndPower

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Класс для выполнения статистического анализа A/B тестов.
    """
    def __init__(self, df: pd.DataFrame):
        if not all(col in df.columns for col in ['test_group', 'converted']):
            raise ValueError("DataFrame должен содержать колонки 'test_group' и 'converted'.")
        self.df = df
        self.results: Dict = {}

    # --- ТРИ МЕТОДА ДЛЯ ПРОВЕРКИ SRM ---

    def validate_srm_chi_square(self, expected_ratios: Dict[str, float] = None) -> Dict:
        """
        Метод 1: Проверка SRM с помощью Chi-square теста.
        """
        group_counts = self.df['test_group'].value_counts()
        total_users = group_counts.sum()

        if expected_ratios is None:
            num_groups = len(group_counts)
            expected_ratios = {group: 1.0 / num_groups for group in group_counts.index}
        
        if not np.isclose(sum(expected_ratios.values()), 1.0):
            raise ValueError("Сумма долей в expected_ratios должна быть 1.0")

        observed = group_counts.loc[list(expected_ratios.keys())].values
        expected = [total_users * ratio for ratio in expected_ratios.values()]
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        
        return {
            'method': 'Chi-square Test',
            'p_value': p_value,
            'has_srm': p_value < 0.05,
            'message': '⚠️ SRM обнаружен' if p_value < 0.05 else '✅ SRM не обнаружен'
        }

    def validate_srm_z_test(self, group: str = 'ad', expected_ratio: float = 0.5) -> Dict:
        """
        Метод 2: Проверка SRM с помощью Z-теста для одной пропорции.
        Сравнивает долю одной группы с ожидаемой.
        """
        total_users = len(self.df)
        group_count = self.df['test_group'].value_counts().get(group, 0)
        
        z_stat, p_value = proportion.proportions_ztest(count=group_count, nobs=total_users, value=expected_ratio)
        
        return {
            'method': f'Z-test for Single Proportion (group {group})',
            'p_value': p_value,
            'has_srm': p_value < 0.05,
            'message': '⚠️ SRM обнаружен' if p_value < 0.05 else '✅ SRM не обнаружен'
        }
        
    def validate_srm_monte_carlo(self, expected_ratios: Dict[str, float] = None, simulations: int = 10000) -> Dict:
        """
        Метод 3: Проверка SRM методом Монте-Карло.
        """
        group_counts = self.df['test_group'].value_counts()
        groups = list(group_counts.index)
        total = group_counts.sum()

        if expected_ratios is None:
            expected_ratios = {g: 1.0 / len(groups) for g in groups}

        expected_props = [expected_ratios[g] for g in groups]
        observed_counts = group_counts.loc[groups].values
        
        expected_chi2_values = np.zeros(simulations)
        for i in range(simulations):
            simulated_counts = np.random.multinomial(total, expected_props)
            expected_chi2_values[i] = stats.chisquare(simulated_counts, observed_counts)[0]
        
        observed_chi2 = stats.chisquare(observed_counts, [p * total for p in expected_props])[0]
        p_value = np.mean(expected_chi2_values >= observed_chi2)
        
        return {
            'method': 'Monte-Carlo Simulation',
            "p_value": p_value,
            "has_srm": p_value < 0.05,
            "message": '⚠️ SRM обнаружен' if p_value < 0.05 else '✅ SRM не обнаружен'
        }

    # --- ТРИ МЕТОДА ДЛЯ СРАВНЕНИЯ КОНВЕРСИЙ ---
    
    def calculate_conversion_rates(self) -> Dict:
        """Расчет конверсии по группам."""
        # ... (код остается без изменений)
        summary = self.df.groupby('test_group')['converted'].agg(['sum', 'count'])
        summary['rate'] = summary['sum'] / summary['count']
        
        conversion_rates = {}
        for group, row in summary.iterrows():
            conversion_rates[group] = {
                'conversions': int(row['sum']),
                'total': int(row['count']),
                'conversion_rate': row['rate']
            }

        if 'ad' in conversion_rates and 'psa' in conversion_rates:
            ad_rate = conversion_rates['ad']['conversion_rate']
            psa_rate = conversion_rates['psa']['conversion_rate']
            conversion_rates['difference'] = {
                'absolute': ad_rate - psa_rate,
                'relative': (ad_rate - psa_rate) / psa_rate * 100 if psa_rate > 0 else float('inf')
            }
        
        self.results['conversion_rates'] = conversion_rates
        return conversion_rates
        
    def perform_z_test_for_proportions(self, group_a: str = 'psa', group_b: str = 'ad') -> Dict:
        """
        Метод 1: Z-тест для разницы двух пропорций.
        """
        if 'conversion_rates' not in self.results: self.calculate_conversion_rates()
        conv_rates = self.results['conversion_rates']
        
        successes = [conv_rates[group_b]['conversions'], conv_rates[group_a]['conversions']]
        nobs = [conv_rates[group_b]['total'], conv_rates[group_a]['total']]

        z_stat, p_value = proportion.proportions_ztest(successes, nobs, alternative='two-sided')
        
        # ... (расчет доверительных интервалов)
        rate_b, rate_a = successes[0] / nobs[0], successes[1] / nobs[1]
        diff = rate_b - rate_a
        se_diff = np.sqrt(rate_b * (1 - rate_b) / nobs[0] + rate_a * (1 - rate_a) / nobs[1])
        ci_diff = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

        return {
            'method': 'Z-test for Proportions',
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'difference_confidence_interval': ci_diff
        }

    def perform_chi_square_test_of_independence(self) -> Dict:
        """
        Метод 2: Хи-квадрат тест независимости.
        Проверяет, есть ли связь между группой и фактом конверсии.
        """
        contingency_table = pd.crosstab(self.df['test_group'], self.df['converted'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        return {
            'method': 'Chi-square Test of Independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_table': contingency_table
        }

    def bootstrap_analysis(self, group_a: str = 'psa', group_b: str = 'ad', n_bootstrap: int = 10000) -> Dict:
        """
        Метод 3: Bootstrap-анализ для оценки распределения разницы конверсий.
        """
        data_a = self.df[self.df['test_group'] == group_a]['converted'].values
        data_b = self.df[self.df['test_group'] == group_b]['converted'].values

        bootstrap_diffs = np.array([
            np.random.choice(data_b, len(data_b), replace=True).mean() - 
            np.random.choice(data_a, len(data_a), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        
        p_value = 2 * min((bootstrap_diffs < 0).mean(), (bootstrap_diffs > 0).mean())
        
        return {
            'method': 'Bootstrap Analysis',
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_difference': bootstrap_diffs.mean(),
            'confidence_interval_95': np.percentile(bootstrap_diffs, [2.5, 97.5]),
            'bootstrap_differences': bootstrap_diffs,
        }

    def generate_summary_report(self) -> Dict:
        """Генерация сводного отчета."""
        # Используем Z-тест как основной для отчета из-за его CI для разницы
        z_test_res = self.perform_z_test_for_proportions()
        conv_rates = self.results['conversion_rates']
        
        summary = {
            'statistical_significance': z_test_res['significant'],
            'p_value': z_test_res['p_value'],
            'absolute_difference': conv_rates.get('difference', {}).get('absolute'),
            'relative_difference': conv_rates.get('difference', {}).get('relative'),
            'confidence_interval_diff': z_test_res['difference_confidence_interval']
        }
        
        if summary['statistical_significance']:
            recommendation = "🚀 РЕКОМЕНДАЦИЯ: Раскатывать." if summary['absolute_difference'] > 0 else "🔴 РЕКОМЕНДАЦИЯ: Откатить."
        else:
            recommendation = "🟡 РЕКОМЕНДАЦИЯ: Оставить без изменений."
        
        summary['recommendation'] = recommendation
        return summary