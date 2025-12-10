import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import proportion
from typing import Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame): 
        self.df = df
        self.results = {}

    def validate_srm(self, expected_ratios: Dict[str, float] = None) -> Dict:
        """
        Проверка Sample Ratio Mismatch (SRM)

        expected_ratios: словарь с ожидаемыми пропорциями для каждой группы,
                        например {'ad': 0.6, 'psa': 0.4}.
                        Сумма должна быть ровно 1.0.
                        По умолчанию 0.5 для каждой группы.
        """
        group_counts = self.df['test_group'].value_counts()
        total = group_counts.sum()

        # Если словарь не задан, используем 0.5 для всех групп
        if expected_ratios is None:
            expected_ratios = {group: 0.5 for group in group_counts.index}

        # Проверка и установка дефолтных значений для отсутствующих групп
        for group in group_counts.index:
            if group not in expected_ratios:
                expected_ratios[group] = 0.5

        # Проверка, что сумма пропорций = 1
        total_ratio = sum(expected_ratios.values())
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Сумма expected_ratios = {total_ratio:.6f}, должна быть ровно 1.0")

        # Ожидаемые counts
        expected_counts = {group: total * expected_ratios[group] for group in group_counts.index}

        # Chi-square test
        chi2, p_value = stats.chisquare(
            list(group_counts.values),
            list(expected_counts.values())
        )

        srm_result = {
            'group_counts': group_counts.to_dict(),
            'expected_counts': expected_counts,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'has_srm': p_value < 0.05,
            'message': '⚠️ Обнаружен SRM! Рандомизация могла быть нарушена' if p_value < 0.05 else '✅ SRM проверка пройдена'
        }

        logger.info(f"SRM проверка: {srm_result['message']}")
        logger.info(f"P-value: {p_value:.6f}")

        return srm_result

    def calculate_conversion_rates(self) -> Dict:
        """Расчет конверсии по группам"""
        conversion_rates = {}

        for group in self.df['test_group'].unique():
            group_data = self.df[self.df['test_group'] == group]
            conversions = group_data['converted'].sum()
            total = len(group_data)
            rate = conversions / total

            conversion_rates[group] = {
                'conversions': conversions,
                'total': total,
                'conversion_rate': rate,
                'conversion_percentage': rate * 100
            }

        # Разница в конверсии
        if 'ad' in conversion_rates and 'psa' in conversion_rates:
            ad_rate = conversion_rates['ad']['conversion_rate']
            psa_rate = conversion_rates['psa']['conversion_rate']
            conversion_rates['difference'] = {
                'absolute': ad_rate - psa_rate,
                'relative': (ad_rate - psa_rate) / psa_rate * 100
            }

        self.results['conversion_rates'] = conversion_rates
        return conversion_rates

    def perform_proportion_test(self) -> Dict:
        """Z-тест для разницы пропорций"""
        conv_rates = self.calculate_conversion_rates()

        if 'ad' not in conv_rates or 'psa' not in conv_rates:
            raise ValueError("Обе группы (ad и psa) должны присутствовать в данных")

        # Данные для теста
        successes = [conv_rates['ad']['conversions'], conv_rates['psa']['conversions']]
        nobs = [conv_rates['ad']['total'], conv_rates['psa']['total']]

        # Z-тест для двух пропорций
        z_stat, p_value = proportion.proportions_ztest(successes, nobs)

        # Доверительные интервалы (95%)
        ad_ci = proportion.proportion_confint(successes[0], nobs[0], alpha=0.05)
        psa_ci = proportion.proportion_confint(successes[1], nobs[1], alpha=0.05)

        # Confidence interval for difference
        diff = conv_rates['difference']['absolute']
        se = np.sqrt(
            conv_rates['ad']['conversion_rate'] * (1 - conv_rates['ad']['conversion_rate']) / nobs[0] +
            conv_rates['psa']['conversion_rate'] * (1 - conv_rates['psa']['conversion_rate']) / nobs[1]
        )
        diff_ci = (diff - 1.96 * se, diff + 1.96 * se)

        result = {
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_level': 0.95,
            'ad_confidence_interval': ad_ci,
            'psa_confidence_interval': psa_ci,
            'difference_confidence_interval': diff_ci
        }

        self.results['proportion_test'] = result
        return result

    def validate_srm_monte_carlo(self, expected_ratios=None, simulations=10000):
        """
        Проверка Sample Ratio Mismatch (SRM) методом Монте-Карло.
        Подходит для любых пропорций и любого количества групп.
        """
        print("🔍 SRM ПРОВЕРКА (метод 4 — Monte-Carlo Simulation)")

        # Реальные данные
        group_counts = self.df['test_group'].value_counts().to_dict()
        groups = list(group_counts.keys())
        total = sum(group_counts.values())

        # Ожидаемые пропорции
        if expected_ratios is None:
            expected_ratios = {g: 1 / len(groups) for g in groups}

        # Проверяем сумму пропорций
        if not np.isclose(sum(expected_ratios.values()), 1.0):
            raise ValueError("Сумма expected_ratios должна быть 1.0")

        # Ожидаемые counts
        expected_counts = {g: total * expected_ratios[g] for g in groups}

        # Встроенная функция для генерирования одной симуляции
        def simulate_once():
            simulated = np.random.multinomial(total, [expected_ratios[g] for g in groups])
            return simulated

        # Фактический вектор
        observed = np.array([group_counts[g] for g in groups])

        # Количество расхождений
        diffs = []
        for _ in range(simulations):
            sim = simulate_once()
            diff = np.sum((sim - observed) ** 2 / expected_counts[g] for sim, g in zip(sim, groups))
            diffs.append(diff)

        diffs = np.array(diffs)

        # Наше наблюдение
        observed_diff = np.sum((observed - np.array(list(expected_counts.values()))) ** 2
                            / np.array(list(expected_counts.values())))

        # P-value
        p_value = np.mean(diffs >= observed_diff)

        message = (
            "⚠️ SRM обнаружен — распределение маловероятно при честной рандомизации."
            if p_value < 0.05
            else "✅ SRM не обнаружен — распределение в пределах нормы."
        )

        return {
            "message": message,
            "p_value": float(p_value),
            "observed_counts": group_counts,
            "expected_counts": expected_counts,
            "simulations": simulations,
            "observed_stat": float(observed_diff),
        }

    def bootstrap_analysis(self, n_bootstrap: int = 10000) -> Dict:
        """Bootstrap анализ для разницы конверсий"""
        ad_data = self.df[self.df['test_group'] == 'ad']['converted']
        psa_data = self.df[self.df['test_group'] == 'psa']['converted']

        bootstrap_differences = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            ad_sample = np.random.choice(ad_data, size=len(ad_data), replace=True)
            psa_sample = np.random.choice(psa_data, size=len(psa_data), replace=True)

            # Difference in conversion rates
            diff = ad_sample.mean() - psa_sample.mean()
            bootstrap_differences.append(diff)

        bootstrap_differences = np.array(bootstrap_differences)

        # Confidence intervals
        ci_95 = np.percentile(bootstrap_differences, [2.5, 97.5])
        ci_90 = np.percentile(bootstrap_differences, [5, 95])

        result = {
            'bootstrap_differences': bootstrap_differences,
            'mean_difference': bootstrap_differences.mean(),
            'confidence_interval_95': ci_95,
            'confidence_interval_90': ci_90,
            'p_value': (bootstrap_differences <= 0).mean()  # one-sided p-value
        }

        self.results['bootstrap'] = result
        return result

    def stratified_analysis(self, stratum_column: str) -> Dict:
        """Стратифицированный анализ по указанной колонке"""
        strata_results = {}

        for stratum in self.df[stratum_column].unique():
            stratum_data = self.df[self.df[stratum_column] == stratum]
            stratum_analyzer = StatisticalAnalyzer(stratum_data)

            strata_results[stratum] = {
                'conversion_rates': stratum_analyzer.calculate_conversion_rates(),
                'sample_size': len(stratum_data)
            }

        self.results[f'stratified_{stratum_column}'] = strata_results
        return strata_results

    def calculate_power(self, alpha: float = 0.05, effect_size: float = None) -> Dict:
        """Расчет мощности теста"""
        from statsmodels.stats.power import NormalIndPower

        if effect_size is None:
            # Используем наблюдаемую разницу
            conv_rates = self.calculate_conversion_rates()
            if 'difference' in conv_rates:
                effect_size = conv_rates['difference']['absolute']
            else:
                effect_size = 0.01  # default MDE

        # Размеры групп
        group_sizes = {
            group: data['total']
            for group, data in self.results['conversion_rates'].items()
            if group in ['ad', 'psa']
        }

        # Расчет мощности
        power_analysis = NormalIndPower()
        power = power_analysis.power(
            effect_size=effect_size,
            nobs1=min(group_sizes.values()),
            alpha=alpha,
            ratio=max(group_sizes.values()) / min(group_sizes.values())
        )

        result = {
            'power': power,
            'effect_size': effect_size,
            'alpha': alpha,
            'min_detectable_effect': effect_size,
            'message': f"Мощность теста: {power:.3f} (>{alpha} - {'✅ Достаточно' if power > alpha else '⚠️ Мало'})"
        }

        self.results['power_analysis'] = result
        return result

    def generate_summary_report(self) -> Dict:
        """Генерация сводного отчета"""
        if not self.results:
            self.calculate_conversion_rates()
            self.perform_proportion_test()

        summary = {
            'sample_sizes': {
                group: data['total']
                for group, data in self.results['conversion_rates'].items()
                if group in ['ad', 'psa']
            },
            'conversion_rates': {
                group: data['conversion_rate']
                for group, data in self.results['conversion_rates'].items()
                if group in ['ad', 'psa']
            },
            'statistical_significance': self.results['proportion_test']['significant'],
            'p_value': self.results['proportion_test']['p_value'],
            'absolute_difference': self.results['conversion_rates']['difference']['absolute'],
            'relative_difference': self.results['conversion_rates']['difference']['relative'],
            'confidence_interval': self.results['proportion_test']['difference_confidence_interval']
        }

        # Рекомендация
        if summary['statistical_significance'] and summary['absolute_difference'] > 0:
            summary['recommendation'] = "🚀 РЕКОМЕНДУЕМ раскат: статистически значимое улучшение"
        elif summary['statistical_significance'] and summary['absolute_difference'] < 0:
            summary['recommendation'] = "🔴 РЕКОМЕНДУЕМ откат: статистически значимое ухудшение"
        else:
            summary['recommendation'] = "🟡 НЕОПРЕДЕЛЕННО: нет статистической значимости"

        return summary