import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import Counter
import math

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas - Corrigido",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# Helpers para multiprocessing
# ----------------------------
def analyze_single_u_alert_id_recurrence(u_alert_id, df_original):
    try:
        df_ci = df_original[df_original['u_alert_id'] == u_alert_id].copy()
        df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
        df_ci = df_ci.dropna(subset=['created_on']).sort_values('created_on')

        if len(df_ci) < 3:
            return {
                'u_alert_id': u_alert_id,
                'total_occurrences': len(df_ci),
                'score': 0,
                'classification': '⚪ DADOS INSUFICIENTES',
                'mean_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'predictability_score': 0
            }

        analyzer = AdvancedRecurrenceAnalyzer(df_ci, u_alert_id)
        return analyzer.analyze_complete_silent()

    except Exception as e:
        return {
            'u_alert_id': u_alert_id,
            'total_occurrences': 0,
            'score': 0,
            'classification': f'⚪ ERRO: {str(e)[:50]}',
            'mean_interval_hours': None,
            'cv': None,
            'regularity_score': 0,
            'periodicity_detected': False,
            'predictability_score': 0
        }


def analyze_chunk_recurrence(u_alert_id_list, df_original):
    results = []
    for u_alert_id in u_alert_id_list:
        result = analyze_single_u_alert_id_recurrence(u_alert_id, df_original)
        if result:
            results.append(result)
    return results


# ============================================================
# AdvancedRecurrenceAnalyzer: análise (UI render opcional)
# ============================================================
class AdvancedRecurrenceAnalyzer:
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id

    def _prepare_data(self):
        if self.df is None or len(self.df) < 3:
            return None
        df = self.df.sort_values('created_on').copy()
        df['created_on'] = pd.to_datetime(df['created_on'], errors='coerce')
        df = df.dropna(subset=['created_on'])
        df['timestamp'] = df['created_on'].astype('int64') // 10**9
        df['time_diff_seconds'] = df['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        dt = df['created_on'].dt
        df['hour'] = dt.hour
        df['day_of_week'] = dt.dayofweek
        df['day_of_month'] = dt.day
        df['week_of_year'] = dt.isocalendar().week
        df['month'] = dt.month
        df['day_name'] = dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] <= 17)
        return df

    # Main public methods
    def analyze(self):
        """Modo interativo (Streamlit)"""
        st.header("🔄 Análise Avançada de Reincidência Temporal")
        df = self._prepare_data()
        if df is None:
            st.warning("⚠️ Dados insuficientes (mínimo 3 ocorrências).")
            return

        st.info(f"📊 Analisando **{len(df)}** ocorrências do Short CI: **{self.alert_id}**")
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            st.warning("⚠️ Intervalos insuficientes.")
            return

        results = {}
        # executar análises com render=True
        results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=True)
        results['regularity'] = self._analyze_regularity(intervals_hours, render=True)
        results['periodicity'] = self._analyze_periodicity(intervals_hours, render=True)
        results['autocorr'] = self._analyze_autocorrelation(intervals_hours, render=True)
        results['temporal'] = self._analyze_temporal_patterns(df, render=True)
        results['clusters'] = self._analyze_clusters(df, intervals_hours, render=True)
        results['bursts'] = self._detect_bursts(intervals_hours, render=True)
        results['seasonality'] = self._analyze_seasonality(df, render=True)
        results['changepoints'] = self._detect_changepoints(intervals_hours, render=True)
        results['anomalies'] = self._detect_anomalies(intervals_hours, render=True)
        results['predictability'] = self._calculate_predictability(intervals_hours, render=True)
        results['stability'] = self._analyze_stability(intervals_hours, df, render=True)
        results['contextual'] = self._analyze_contextual_dependencies(df, render=True)
        results['vulnerability'] = self._identify_vulnerability_windows(df, intervals_hours, render=True)
        results['maturity'] = self._analyze_pattern_maturity(df, intervals_hours, render=True)
        results['prediction_confidence'] = self._calculate_prediction_confidence(intervals_hours, render=True)
        results['markov'] = self._analyze_markov_chains(intervals_hours, render=True)
        results['randomness'] = self._advanced_randomness_tests(intervals_hours, render=True)

        self._final_classification(results, df, intervals_hours)

    def analyze_complete_silent(self):
        """Modo silencioso para batch: retorna dict resumo"""
        df = self._prepare_data()
        if df is None or len(df) < 3:
            return None
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            return None

        results = {}
        # executar análises com render=False (silencioso)
        try:
            results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=False)
        except Exception:
            results['basic_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'cv': 0}

        try:
            results['regularity'] = self._analyze_regularity(intervals_hours, render=False)
        except Exception:
            results['regularity'] = {'cv': 0, 'regularity_score': 0}

        try:
            results['periodicity'] = self._analyze_periodicity(intervals_hours, render=False)
        except Exception:
            results['periodicity'] = {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}

        try:
            results['predictability'] = self._calculate_predictability(intervals_hours, render=False)
        except Exception:
            results['predictability'] = {'predictability_score': 0, 'next_expected_hours': 0}

        try:
            results['temporal'] = self._analyze_temporal_patterns(df, render=False)
        except Exception:
            results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}

        # calcular score final
        final_score, classification = self._calculate_final_score_validated(results, df, intervals_hours)

        return {
            'u_alert_id': self.alert_id,
            'total_occurrences': len(df),
            'score': final_score,
            'classification': classification,
            'mean_interval_hours': results['basic_stats'].get('mean'),
            'median_interval_hours': results['basic_stats'].get('median'),
            'cv': results['basic_stats'].get('cv'),
            'regularity_score': results['regularity'].get('regularity_score'),
            'periodicity_detected': results['periodicity'].get('has_strong_periodicity', False),
            'dominant_period_hours': results['periodicity'].get('dominant_period_hours'),
            'predictability_score': results['predictability'].get('predictability_score'),
            'next_occurrence_prediction_hours': results['predictability'].get('next_expected_hours'),
            'hourly_concentration': results['temporal'].get('hourly_concentration'),
            'daily_concentration': results['temporal'].get('daily_concentration'),
        }

    # ----------------------------
    # Métodos unificados (render opcional)
    # ----------------------------
    def _analyze_basic_statistics(self, intervals, render=True):
        stats_dict = {
            'mean': float(np.mean(intervals)),
            'median': float(np.median(intervals)),
            'std': float(np.std(intervals)),
            'min': float(np.min(intervals)),
            'max': float(np.max(intervals)),
            'cv': float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')),
            'q25': float(np.percentile(intervals, 25)),
            'q75': float(np.percentile(intervals, 75)),
            'iqr': float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
        }
        if render:
            st.subheader("📊 1. Estatísticas de Intervalos")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
            col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
            col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
            col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
            col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
        return stats_dict

    def _analyze_regularity(self, intervals, render=True):
        cv = float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'))
        if cv < 0.20:
            regularity_score, pattern_type, pattern_color = 95, "🟢 ALTAMENTE REGULAR", "green"
        elif cv < 0.40:
            regularity_score, pattern_type, pattern_color = 80, "🟢 REGULAR", "lightgreen"
        elif cv < 0.70:
            regularity_score, pattern_type, pattern_color = 60, "🟡 SEMI-REGULAR", "yellow"
        elif cv < 1.20:
            regularity_score, pattern_type, pattern_color = 35, "🟠 IRREGULAR", "orange"
        else:
            regularity_score, pattern_type, pattern_color = 15, "🔴 MUITO IRREGULAR", "red"

        if render:
            st.subheader("🎯 2. Regularidade")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Classificação:** {pattern_type}")
                st.write(f"**CV:** {cv:.2%}")
                if len(intervals) >= 3:
                    _, p_value = stats.shapiro(intervals)
                    if p_value > 0.05:
                        st.info("📊 Distribuição aproximadamente normal")
                    else:
                        st.warning("📊 Distribuição não-normal")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=regularity_score,
                    title={'text': "Regularidade"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': pattern_color}}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'reg_gauge_{self.alert_id}')
        return {'cv': cv, 'regularity_score': regularity_score, 'type': pattern_type}

    def _analyze_periodicity(self, intervals, render=True):
        if len(intervals) < 10:
            if render:
                st.subheader("🔍 3. Periodicidade (FFT)")
                st.info("📊 Mínimo de 10 intervalos necessários")
            return {'periods': [], 'has_periodicity': False, 'has_strong_periodicity': False, 'dominant_period_hours': None}

        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
        intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
        fft_vals = fft(intervals_padded)
        freqs = fftfreq(n_padded, d=1)
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        fft_mag = np.abs(fft_vals[positive_idx])
        threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
        peaks_idx = fft_mag > threshold

        dominant_periods = []
        has_strong_periodicity = False
        dominant_period_hours = None
        if np.any(peaks_idx):
            dominant_freqs = freqs_pos[peaks_idx]
            dominant_periods = (1 / dominant_freqs)
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            if len(dominant_periods) > 0:
                has_strong_periodicity = True
                dominant_period_hours = float(dominant_periods[0] * np.mean(intervals))

        if render:
            st.subheader("🔍 3. Periodicidade (FFT)")
            if has_strong_periodicity:
                st.success("🎯 **Periodicidades Detectadas:**")
                for period in dominant_periods:
                    est_time = period * np.mean(intervals)
                    time_str = f"{est_time:.1f}h" if est_time < 24 else f"{est_time/24:.1f} dias"
                    st.write(f"• Período: **{period:.1f}** ocorrências (~{time_str})")
            else:
                st.info("📊 Nenhuma periodicidade forte detectada")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=1/freqs_pos[:len(freqs_pos)//4],
                y=fft_mag[:len(freqs_pos)//4],
                mode='lines',
                fill='tozeroy'
            ))
            fig.update_layout(title="Espectro de Frequência", xaxis_title="Período", yaxis_title="Magnitude", height=300, xaxis_type="log")
            st.plotly_chart(fig, use_container_width=True, key=f'fft_{self.alert_id}')

        return {'periods': list(map(float, dominant_periods)) if len(dominant_periods) else [], 'has_periodicity': len(dominant_periods) > 0, 'has_strong_periodicity': has_strong_periodicity, 'dominant_period_hours': dominant_period_hours}

    def _analyze_autocorrelation(self, intervals, render=True):
        if len(intervals) < 5:
            if render:
                st.subheader("📈 4. Autocorrelação")
                st.info("Insuficiente para autocorrelação")
            return {'peaks': [], 'has_autocorr': False, 'max_autocorr': 0}

        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        lags = np.arange(len(autocorr))
        threshold = 2 / np.sqrt(len(intervals))
        significant_peaks = [(i, float(autocorr[i])) for i in range(1, min(len(autocorr), 20)) if autocorr[i] > threshold]
        max_autocorr = max([corr for _, corr in significant_peaks], default=0)

        if render:
            st.subheader("📈 4. Autocorrelação")
            if significant_peaks:
                st.success("✅ **Autocorrelação Significativa:**")
                for lag, corr in significant_peaks[:3]:
                    st.write(f"• Lag {lag}: {corr:.2f}")
            else:
                st.info("📊 Sem autocorrelação significativa")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lags[:min(30, len(lags))], y=autocorr[:min(30, len(autocorr))], mode='lines+markers'))
            fig.add_hline(y=threshold, line_dash="dash", line_color="red")
            fig.add_hline(y=-threshold, line_dash="dash", line_color="red")
            fig.update_layout(title="Autocorrelação", height=300)
            st.plotly_chart(fig, use_container_width=True, key=f'autocorr_{self.alert_id}')

        return {'peaks': significant_peaks, 'has_autocorr': len(significant_peaks) > 0, 'max_autocorr': max_autocorr}

    def _analyze_temporal_patterns(self, df, render=True):
        hourly = df.groupby('hour').size().reindex(range(24), fill_value=0)
        daily = df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
        hourly_conc = float(hourly_pct.nlargest(3).sum()) if len(hourly_pct) > 0 else 0.0
        daily_conc = float(daily_pct.nlargest(3).sum()) if len(daily_pct) > 0 else 0.0
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist() if len(hourly) > 0 else []
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist() if len(daily) > 0 else []

        if render:
            st.subheader("⏰ 5. Padrões Temporais")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Bar(x=list(range(24)), y=hourly.values, marker_color=['red' if v > hourly.mean() + hourly.std() else 'lightblue' for v in hourly.values]))
                fig.update_layout(title="Por Hora", xaxis_title="Hora", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'hourly_{self.alert_id}')
                if peak_hours:
                    st.success(f"🕐 **Picos:** {', '.join([f'{h:02d}:00' for h in peak_hours])}")
            with col2:
                days_map = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                fig = go.Figure(go.Bar(x=days_map, y=daily.values, marker_color=['red' if v > daily.mean() + daily.std() else 'lightgreen' for v in daily.values]))
                fig.update_layout(title="Por Dia", xaxis_title="Dia", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'daily_{self.alert_id}')
                if peak_days:
                    st.success(f"📅 **Picos:** {', '.join([days_map[d] for d in peak_days])}")

        return {'hourly_concentration': hourly_conc, 'daily_concentration': daily_conc, 'peak_hours': peak_hours, 'peak_days': peak_days}

    def _analyze_clusters(self, df, intervals, render=True):
        if len(df) < 10:
            if render:
                st.subheader("🎯 6. Clusters Temporais")
                st.info("Mínimo de 10 ocorrências necessário")
            return {'n_clusters': 0, 'n_noise': 0}

        first_ts = df['timestamp'].min()
        time_features = ((df['timestamp'] - first_ts) / 3600).values.reshape(-1, 1)
        eps = float(np.median(intervals) * 2) if len(intervals) > 0 else 1.0
        clusters = DBSCAN(eps=eps, min_samples=3).fit_predict(time_features)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)

        if render:
            st.subheader("🎯 6. Clusters Temporais")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Clusters", n_clusters)
            col2.metric("📊 Em Clusters", len(clusters) - n_noise)
            col3.metric("🔴 Isolados", n_noise)
            if n_clusters > 0:
                st.success(f"✅ **{n_clusters} clusters** identificados")

        return {'n_clusters': int(n_clusters), 'n_noise': int(n_noise)}

    def _detect_bursts(self, intervals, render=True):
        if len(intervals) < 3:
            if render:
                st.subheader("💥 7. Detecção de Bursts")
                st.info("Insuficiente para detectar bursts")
            return {'n_bursts': 0, 'has_bursts': False}

        burst_threshold = np.percentile(intervals, 25)
        is_burst = intervals < burst_threshold
        burst_changes = np.diff(np.concatenate(([False], is_burst, [False])))
        burst_starts = np.where(burst_changes == 1)[0]
        burst_ends = np.where(burst_changes == -1)[0]
        burst_sequences = [(int(start), int(end)) for start, end in zip(burst_starts, burst_ends) if end - start >= 3]

        if render:
            st.subheader("💥 7. Detecção de Bursts")
            col1, col2 = st.columns(2)
            col1.metric("💥 Bursts", len(burst_sequences))
            if burst_sequences:
                avg_size = np.mean([end - start for start, end in burst_sequences])
                col2.metric("📊 Tamanho Médio", f"{avg_size:.1f}")
                st.warning(f"⚠️ **{len(burst_sequences)} bursts** detectados")
            else:
                st.success("✅ Sem padrão de rajadas")

        return {'n_bursts': int(len(burst_sequences)), 'has_bursts': len(burst_sequences) > 0}

    def _analyze_seasonality(self, df, render=True):
        date_range = (df['created_on'].max() - df['created_on'].min()).days
        if render:
            st.subheader("🌡️ 8. Sazonalidade")
        if date_range < 30:
            if render:
                st.info("📊 Período curto para análise sazonal")
            return {'trend': 'stable'}

        weekly = df.groupby('week_of_year').size()
        if len(weekly) >= 4 and render:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly.values, mode='lines+markers', fill='tozeroy'))
            fig.update_layout(title="Evolução Semanal", height=250)
            st.plotly_chart(fig, use_container_width=True, key=f'weekly_{self.alert_id}')
            if len(weekly) > 3:
                slope, _, _, p_value, _ = stats.linregress(weekly.index.values, weekly.values)
                if p_value < 0.05:
                    if slope > 0:
                        st.warning("📈 **Tendência crescente**")
                        return {'trend': 'increasing', 'slope': float(slope)}
                    else:
                        st.success("📉 **Tendência decrescente**")
                        return {'trend': 'decreasing', 'slope': float(slope)}
        return {'trend': 'stable'}

    def _detect_changepoints(self, intervals, render=True):
        if len(intervals) < 20:
            if render:
                st.subheader("🔀 9. Pontos de Mudança")
                st.info("Mínimo de 20 intervalos necessário")
            return {'changepoints': [], 'has_changes': False}

        cumsum = np.cumsum(intervals - np.mean(intervals))
        window = 5
        changes = []
        for i in range(window, len(cumsum) - window):
            before = np.mean(intervals[max(0, i - window):i])
            after = np.mean(intervals[i:min(len(intervals), i + window)])
            if abs(before - after) > np.std(intervals):
                changes.append(int(i))

        filtered = []
        for cp in changes:
            if not filtered or cp - filtered[-1] > 5:
                filtered.append(cp)

        if render:
            st.subheader("🔀 9. Pontos de Mudança")
            if filtered:
                st.warning(f"⚠️ **{len(filtered)} pontos de mudança** detectados")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(cumsum))), y=cumsum, mode='lines'))
                for cp in filtered:
                    fig.add_vline(x=cp, line_dash="dash", line_color="red")
                fig.update_layout(title="CUSUM", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'cusum_{self.alert_id}')
            else:
                st.success("✅ Comportamento estável")

        return {'changepoints': filtered, 'has_changes': len(filtered) > 0}

    def _detect_anomalies(self, intervals, render=True):
        if len(intervals) == 0:
            return {'anomaly_rate': 0.0, 'total_anomalies': 0}

        z_scores = np.abs(stats.zscore(intervals))
        z_anomalies = int(np.sum(z_scores > 3))
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_anomalies = int(np.sum((intervals < lower) | (intervals > upper)))

        iso_anomalies = 0
        if len(intervals) >= 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(intervals.reshape(-1, 1))
            iso_anomalies = int(np.sum(predictions == -1))

        total_anomalies = max(z_anomalies, iqr_anomalies, iso_anomalies)
        anomaly_rate = float(total_anomalies / len(intervals) * 100)

        if render:
            st.subheader("🚨 10. Detecção de Anomalias")
            col1, col2, col3 = st.columns(3)
            col1.metric("Z-Score", f"{z_anomalies}")
            col2.metric("IQR", f"{iqr_anomalies}")
            col3.metric("Iso. Forest", f"{iso_anomalies}")
            if anomaly_rate > 10:
                st.warning(f"⚠️ **{anomaly_rate:.1f}%** de anomalias")
            else:
                st.success("✅ Baixa taxa de anomalias")

        return {'anomaly_rate': anomaly_rate, 'total_anomalies': total_anomalies}

    def _calculate_predictability(self, intervals, render=True):
        cv = float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'))
        if cv < 0.20:
            predictability = 95
        elif cv < 0.40:
            predictability = 80
        elif cv < 0.70:
            predictability = 55
        elif cv < 1.20:
            predictability = 30
        else:
            predictability = 10
        mean_interval = float(np.mean(intervals))

        if render:
            st.subheader("🔮 11. Previsibilidade")
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{predictability}%")
            col2.metric("Próxima Ocorrência", f"{mean_interval:.1f}h")
            if predictability > 70:
                st.success("✅ Altamente previsível")
            elif predictability > 50:
                st.info("📊 Moderadamente previsível")
            else:
                st.warning("⚠️ Pouco previsível")

        return {'predictability_score': int(predictability), 'next_expected_hours': mean_interval}

    def _analyze_stability(self, intervals, df, render=True):
        if len(intervals) < 10:
            return {'is_stable': True, 'stability_score': 50, 'drift_pct': 0.0}
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]
        _, p_value = stats.ttest_ind(first_half, second_half)
        is_stable = p_value > 0.05
        mean_diff = abs(np.mean(second_half) - np.mean(first_half))
        drift_pct = float((mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0)
        stability_score = float(max(0, 100 - drift_pct))

        if render:
            st.subheader("📊 12. Estabilidade")
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{stability_score:.1f}%")
            col2.metric("Drift", f"{drift_pct:.1f}%")
            if is_stable and drift_pct < 20:
                st.success("✅ Padrão estável")
            elif drift_pct < 50:
                st.info("📊 Moderadamente estável")
            else:
                st.warning("⚠️ Padrão instável")

        return {'is_stable': bool(is_stable), 'stability_score': stability_score, 'drift_pct': drift_pct}

    def _analyze_contextual_dependencies(self, df, render=True):
        try:
            years = df['created_on'].dt.year.unique()
            br_holidays = holidays.Brazil(years=years)
            df['is_holiday'] = df['created_on'].dt.date.apply(lambda x: x in br_holidays)
        except Exception:
            df['is_holiday'] = False

        business_days = df[~df['is_weekend'] & ~df['is_holiday']]
        weekend_days = df[df['is_weekend']]
        holiday_days = df[df['is_holiday']]

        if render:
            st.subheader("🌐 13. Dependências Contextuais")
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Dias Úteis", f"{len(business_days)/len(df)*100:.1f}%")
            col2.metric("🎉 Fins de Semana", f"{len(weekend_days)/len(df)*100:.1f}%")
            col3.metric("🎊 Feriados", f"{len(holiday_days)/len(df)*100:.1f}%")
            if len(holiday_days) > 0:
                st.warning(f"⚠️ {len(holiday_days)} alertas em feriados")

        return {'holiday_correlation': float(len(holiday_days) / len(df) if len(df) > 0 else 0), 'weekend_correlation': float(len(weekend_days) / len(df) if len(df) > 0 else 0)}

    def _identify_vulnerability_windows(self, df, intervals, render=True):
        vulnerability_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        if vulnerability_matrix.empty:
            return {'top_windows': []}
        vulnerability_matrix['risk_score'] = (vulnerability_matrix['count'] / vulnerability_matrix['count'].max() * 100)
        top_windows = vulnerability_matrix.nlargest(5, 'risk_score')
        day_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        if render:
            st.subheader("🎯 14. Janelas de Vulnerabilidade")
            st.write("**🔴 Top 5 Janelas Críticas:**")
            for idx, row in top_windows.iterrows():
                day = day_map[row['day_of_week']]
                hour = int(row['hour'])
                risk = row['risk_score']
                st.write(f"• **{day} {hour:02d}:00** - Score: {risk:.1f} ({row['count']} alertas)")
        return {'top_windows': top_windows.to_dict('records')}

    def _analyze_pattern_maturity(self, df, intervals, render=True):
        n_periods = 4
        period_size = len(intervals) // n_periods
        if period_size < 2:
            if render:
                st.subheader("📈 15. Maturidade do Padrão")
                st.info("Período insuficiente")
            return {'maturity': 'stable', 'slope': 0.0}

        periods_stats = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(intervals)
            period_intervals = intervals[start:end]
            periods_stats.append({'period': i + 1, 'mean': float(np.mean(period_intervals)), 'cv': float(np.std(period_intervals) / np.mean(period_intervals) if np.mean(period_intervals) > 0 else 0)})

        periods_df = pd.DataFrame(periods_stats)
        slope = float(np.polyfit(periods_df['period'], periods_df['cv'], 1)[0])

        if render:
            st.subheader("📈 15. Maturidade do Padrão")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=periods_df['period'], y=periods_df['cv'], mode='lines+markers', name='CV', line=dict(color='red', width=3)))
            fig.update_layout(title="Evolução da Variabilidade", xaxis_title="Período", yaxis_title="CV", height=300)
            st.plotly_chart(fig, use_container_width=True, key=f'maturity_{self.alert_id}')
            if slope < -0.05:
                st.success("✅ **Amadurecendo**: Variabilidade decrescente")
                maturity = "maturing"
            elif slope > 0.05:
                st.warning("⚠️ **Degradando**: Variabilidade crescente")
                maturity = "degrading"
            else:
                st.info("📊 **Estável**: Variabilidade constante")
                maturity = "stable"
        else:
            maturity = "maturing" if slope < -0.05 else ("degrading" if slope > 0.05 else "stable")

        return {'maturity': maturity, 'slope': slope}

    def _calculate_prediction_confidence(self, intervals, render=True):
        if len(intervals) < 10:
            return {'confidence': 'low', 'score': 0}
        cv = float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'))
        n_samples = len(intervals)
        regularity_score = max(0, 100 - cv * 100)
        sample_score = min(100, (n_samples / 50) * 100)
        mid = len(intervals) // 2
        var1 = np.var(intervals[:mid])
        var2 = np.var(intervals[mid:])
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0
        stationarity_score = var_ratio * 100
        confidence_score = (regularity_score * 0.5 + sample_score * 0.3 + stationarity_score * 0.2)
        confidence = 'high' if confidence_score > 70 else ('medium' if confidence_score > 40 else 'low')

        if render:
            st.subheader("🎯 16. Confiança de Predição")
            col1, col2 = st.columns(2)
            col1.metric("Confiança", confidence.upper())
            col2.metric("Score", f"{confidence_score:.1f}%")

        return {'confidence': confidence, 'score': float(confidence_score)}

    def _analyze_markov_chains(self, intervals, render=True):
        if len(intervals) < 20:
            if render:
                st.subheader("🔗 17. Cadeias de Markov")
                st.info("Mínimo de 20 intervalos necessário")
            return {'markov_score': 0.0}
        q25, q50, q75 = np.percentile(intervals, [25, 50, 75])
        def interval_to_state(val):
            if val <= q25:
                return 0
            elif val <= q50:
                return 1
            elif val <= q75:
                return 2
            else:
                return 3
        states = [interval_to_state(i) for i in intervals]
        n_states = 4
        transition_matrix = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_matrix[from_state, to_state] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transition_matrix / row_sums
        max_probs = transition_probs.max(axis=1)
        markov_score = float(np.mean(max_probs) * 100)

        if render:
            st.subheader("🔗 17. Cadeias de Markov")
            state_labels = ['Muito Curto', 'Curto', 'Normal', 'Longo']
            fig = go.Figure(data=go.Heatmap(z=transition_probs, x=state_labels, y=state_labels, text=np.round(transition_probs, 2), texttemplate='%{text:.2f}', colorscale='Blues'))
            fig.update_layout(title="Matriz de Transição", xaxis_title="Estado Seguinte", yaxis_title="Estado Atual", height=400)
            st.plotly_chart(fig, use_container_width=True, key=f'markov_matrix_{self.alert_id}')
            st.metric("Score Markoviano", f"{markov_score:.1f}%")
            if markov_score > 60:
                st.success("✅ Forte padrão markoviano")
            elif markov_score > 30:
                st.info("📊 Padrão moderado")
            else:
                st.warning("⚠️ Padrão fraco")

        return {'markov_score': markov_score}

    def _advanced_randomness_tests(self, intervals, render=True):
        if len(intervals) < 10:
            if render:
                st.subheader("🎲 18. Testes de Aleatoriedade")
                st.info("Mínimo de 10 intervalos necessário")
            return {'overall_randomness_score': 50}

        if render:
            st.subheader("🎲 18. Testes de Aleatoriedade")
            st.write("**1️⃣ Runs Test**")
        median = np.median(intervals)
        runs = np.diff(intervals > median).sum() + 1
        expected_runs = len(intervals) / 2
        if render:
            col1, col2 = st.columns(2)
            col1.metric("Runs Observados", int(runs))
            col2.metric("Runs Esperados", f"{expected_runs:.1f}")

        # Permutation entropy
        def permutation_entropy(series, order=3):
            n = len(series)
            permutations = []
            for i in range(n - order + 1):
                pattern = series[i:i+order]
                sorted_idx = np.argsort(pattern)
                perm = tuple(sorted_idx)
                permutations.append(perm)
            perm_counts = Counter(permutations)
            probs = np.array(list(perm_counts.values())) / len(permutations) if len(permutations) > 0 else np.array([1.0])
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(math.factorial(order))
            return entropy / max_entropy if max_entropy > 0 else 0

        perm_entropy = permutation_entropy(intervals)
        complexity = float(perm_entropy * 100)
        if render:
            st.write("**2️⃣ Permutation Entropy**")
            col1, col2 = st.columns(2)
            col1.metric("Entropia", f"{perm_entropy:.3f}")
            col2.metric("Complexidade", f"{complexity:.1f}%")
            if complexity > 70:
                st.success("✅ Alta complexidade")
            else:
                st.warning("⚠️ Baixa complexidade")

        # Hurst
        def hurst_exponent(series):
            n = len(series)
            if n < 20:
                return None
            lags = range(2, min(n//2, 20))
            tau = []
            for lag in lags:
                n_partitions = n // lag
                partitions = [series[i*lag:(i+1)*lag] for i in range(n_partitions)]
                rs_values = []
                for partition in partitions:
                    if len(partition) == 0:
                        continue
                    mean = np.mean(partition)
                    cumsum = np.cumsum(partition - mean)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(partition)
                    if S > 0:
                        rs_values.append(R / S)
                if rs_values:
                    tau.append(np.mean(rs_values))
            if len(tau) > 2:
                log_lags = np.log(list(lags[:len(tau)]))
                log_tau = np.log(tau)
                hurst = np.polyfit(log_lags, log_tau, 1)[0]
                return hurst
            return None

        hurst = hurst_exponent(intervals) if len(intervals) >= 20 else None
        if hurst is not None and render:
            st.write("**3️⃣ Hurst Exponent**")
            st.metric("Hurst", f"{hurst:.3f}")
            if hurst < 0.45:
                st.info("📉 Anti-persistente")
            elif hurst > 0.55:
                st.warning("📈 Persistente")
            else:
                st.success("🎲 Random Walk")

        randomness_score = 50  # simplificado
        if render:
            st.markdown("---")
            st.metric("Score de Aleatoriedade", f"{randomness_score:.0f}%")
            if randomness_score >= 60:
                st.success("✅ Comportamento aleatório")
            elif randomness_score >= 40:
                st.info("📊 Comportamento misto")
            else:
                st.warning("⚠️ Comportamento determinístico")

        return {'overall_randomness_score': randomness_score, 'hurst': hurst, 'perm_entropy': perm_entropy}

    # ----------------------------
    # Classificação final (interna)
    # ----------------------------
    def _calculate_final_score_validated(self, results, df, intervals):
        # 1. Regularidade (20%)
        regularity_score = results['regularity']['regularity_score'] * 0.20

        # 2. Periodicidade (20%)
        if results['periodicity'].get('has_strong_periodicity', False):
            periodicity_score = 100 * 0.20
        elif results['periodicity'].get('has_moderate_periodicity', False):
            periodicity_score = 50 * 0.20
        else:
            periodicity_score = 0 * 0.20

        # 3. Previsibilidade (15%)
        predictability_score = results['predictability']['predictability_score'] * 0.15

        # 4. Concentração Temporal (20%)
        hourly_conc = results['temporal']['hourly_concentration']
        daily_conc = results['temporal']['daily_concentration']
        concentration_score = 0
        if hourly_conc > 60 or daily_conc > 60:
            concentration_score = 100 * 0.20
        elif hourly_conc > 40 or daily_conc > 40:
            concentration_score = 60 * 0.20
        elif hourly_conc > 30 or daily_conc > 30:
            concentration_score = 30 * 0.20

        # 5. Frequência Absoluta (15%)
        total_occurrences = len(df)
        period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
        freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0

        if freq_per_week >= 3:
            frequency_score = 100 * 0.15
        elif freq_per_week >= 1:
            frequency_score = 70 * 0.15
        elif freq_per_week >= 0.5:
            frequency_score = 40 * 0.15
        elif total_occurrences >= 10:
            frequency_score = 30 * 0.15
        else:
            frequency_score = 10 * 0.15

        final_score = (regularity_score + periodicity_score + predictability_score + concentration_score + frequency_score)

        if final_score >= 70 and total_occurrences >= 10:
            classification = "🔴 REINCIDENTE CRÍTICO (P1)"
        elif final_score >= 50 and total_occurrences >= 5:
            classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
        elif final_score >= 35:
            classification = "🟡 PADRÃO DETECTÁVEL (P3)"
        else:
            classification = "🟢 NÃO REINCIDENTE (P4)"

        return round(float(final_score), 2), classification

    def _final_classification(self, results, df, intervals):
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL")
        final_score, classification = self._calculate_final_score_validated(results, df, intervals)

        if final_score >= 70:
            level, color, priority,  = "CRÍTICO", "red", "P1"
        elif final_score >= 50:
            level, color, priority,  = "ALTO", "orange", "P2"
        elif final_score >= 35:
            level, color, priority,  = "MÉDIO", "yellow", "P3" 
        else:
            level, color, priority,  = "BAIXO", "green", "P4"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
            st.metric("Score de Reincidência", f"{final_score:.0f}/100")
            st.markdown("#### 📊 Breakdown dos Critérios VALIDADOS")
            total_occurrences = len(df)
            period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
            freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
            regularity_pts = results['regularity']['regularity_score'] * 0.20
            periodicity_pts = (100 * 0.20) if results['periodicity'].get('has_strong_periodicity', False) else 0
            predictability_pts = results['predictability']['predictability_score'] * 0.15
            hourly_conc = results['temporal']['hourly_concentration']
            daily_conc = results['temporal']['daily_concentration']
            if hourly_conc > 60 or daily_conc > 60:
                concentration_pts = 100 * 0.20
            elif hourly_conc > 40 or daily_conc > 40:
                concentration_pts = 60 * 0.20
            else:
                concentration_pts = 30 * 0.20 if (hourly_conc > 30 or daily_conc > 30) else 0
            if freq_per_week >= 3:
                frequency_pts = 100 * 0.15
            elif freq_per_week >= 1:
                frequency_pts = 70 * 0.15
            else:
                frequency_pts = 40 * 0.15 if freq_per_week >= 0.5 else 10 * 0.15

            breakdown = {
                '1. Regularidade (20%)': regularity_pts,
                '2. Periodicidade (20%)': periodicity_pts,
                '3. Previsibilidade (15%)': predictability_pts,
                '4. Concentração Temporal (20%)': concentration_pts,
                '5. Frequência Absoluta (15%)': frequency_pts,
            }

            for criterion, points in breakdown.items():
                st.write(f"• {criterion}: **{points:.1f} pts**")


        with col2:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=final_score, title={'text': "Score Final"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key=f'final_gauge_{self.alert_id}')

        # Exportar resumo
        st.markdown("---")
        export_data = {
            'u_alert_id': self.alert_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': final_score,
            'classificacao': classification,
            'nivel': level,
            'prioridade': priority,
            'total_occurrences': len(df),
            'freq_per_week': freq_per_week,
            'cv': results['basic_stats']['cv'],
            'regularidade': results['regularity']['regularity_score'],
            'periodicidade': results['periodicity'].get('has_strong_periodicity', False),
            'previsibilidade': results['predictability']['predictability_score'],
            'concentracao_horaria': results['temporal']['hourly_concentration'],
            'concentracao_diaria': results['temporal']['daily_concentration'],
            'bursts_detected': results['bursts']['has_bursts'],
            'n_bursts': results['bursts']['n_bursts'],
        }
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        st.download_button("⬇️ Exportar Relatório Completo", csv, f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)


# ============================================================
# StreamlitAlertAnalyzer: UI glue + batch processing
# ============================================================
class StreamlitAlertAnalyzer:
    def __init__(self):
        self.df_original = None
        self.df = None
        self.dates = None
        self.alert_id = None

    def load_data(self, uploaded_file):
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado: {len(df_raw)} registros")
            with st.expander("📋 Preview"):
                st.write(f"**Colunas:** {list(df_raw.columns)}")
                st.dataframe(df_raw.head())
            if 'created_on' not in df_raw.columns or 'u_alert_id' not in df_raw.columns:
                st.error("❌ Colunas obrigatórias: 'created_on' e 'u_alert_id'")
                return False
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on']).sort_values(['u_alert_id', 'created_on']).reset_index(drop=True)
            self.df_original = df_raw
            st.sidebar.write(f"**IDs:** {len(df_raw['u_alert_id'].unique())}")
            return True
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            return False

    def prepare_individual_analysis(self, alert_id):
        df_filtered = self.df_original[self.df_original['u_alert_id'] == alert_id].copy()
        if len(df_filtered) == 0:
            return False
        df_filtered['date'] = df_filtered['created_on'].dt.date
        df_filtered['hour'] = df_filtered['created_on'].dt.hour
        df_filtered['day_of_week'] = df_filtered['created_on'].dt.dayofweek
        df_filtered['day_name'] = df_filtered['created_on'].dt.day_name()
        df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])
        df_filtered['is_business_hours'] = (df_filtered['hour'] >= 9) & (df_filtered['hour'] <= 17)
        df_filtered['time_diff_hours'] = df_filtered['created_on'].diff().dt.total_seconds() / 3600
        self.df = df_filtered
        self.dates = df_filtered['created_on']
        self.alert_id = alert_id
        return True

    def complete_analysis_all_u_alert_id(self, progress_bar=None):
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None
            u_alert_id_list = list(self.df_original['u_alert_id'].unique())
            total = len(u_alert_id_list)
            use_mp = total > 20
            if use_mp:
                n_processes = min(cpu_count(), total, 8)
                st.info(f"🚀 Usando {n_processes} processos para {total} alertas")
                chunk_size = max(1, total // n_processes)
                chunks = [u_alert_id_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
                process_func = partial(analyze_chunk_recurrence, df_original=self.df_original)
                try:
                    all_results = []
                    with Pool(processes=n_processes) as pool:
                        for idx, chunk_results in enumerate(pool.imap(process_func, chunks)):
                            all_results.extend(chunk_results)
                            if progress_bar:
                                progress = (len(all_results) / total)
                                progress_bar.progress(progress, text=f"{len(all_results)}/{total}")
                    df_results = pd.DataFrame(all_results)
                    if progress_bar:
                        progress_bar.progress(1.0, text="✅ Completa!")
                    return df_results
                except Exception as e:
                    st.warning(f"⚠️ Erro no multiprocessing: {e}. Usando modo sequencial...")
                    use_mp = False
            if not use_mp:
                all_results = []
                for idx, u_alert_id in enumerate(u_alert_id_list):
                    if progress_bar:
                        progress_bar.progress((idx + 1) / total, text=f"{idx + 1}/{total}")
                    result = analyze_single_u_alert_id_recurrence(u_alert_id, self.df_original)
                    if result:
                        all_results.append(result)
                return pd.DataFrame(all_results)
        except Exception as e:
            st.error(f"Erro: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def show_basic_stats(self):
        st.header("📊 Estatísticas Básicas")
        total = len(self.df)
        period_days = (self.dates.max() - self.dates.min()).days + 1
        avg_per_day = total / period_days if period_days > 0 else 0
        unique_days = self.df['date'].nunique()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔥 Total", total)
        col2.metric("📅 Período", period_days)
        col3.metric("📆 Dias Únicos", unique_days)
        col4.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        col5.metric("🕐 Último", self.dates.max().strftime("%d/%m %H:%M"))
        if unique_days == 1:
            st.warning("⚠️ Todos em 1 dia - Pode não ser reincidente")
        st.markdown("---")
        st.subheader("📊 Frequências")
        total_hours = period_days * 24
        avg_per_hour = total / total_hours if total_hours > 0 else 0
        avg_per_week = total / (period_days / 7) if period_days > 0 else 0
        avg_per_month = total / (period_days / 30.44) if period_days > 0 else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Por Dia", f"{avg_per_day:.2f}")
        col2.metric("Por Hora", f"{avg_per_hour:.4f}")
        col3.metric("Por Semana", f"{avg_per_week:.2f}")
        col4.metric("Por Mês", f"{avg_per_month:.2f}")
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            st.markdown("---")
            st.subheader("⏱️ Intervalos")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Média (h)", f"{intervals.mean():.2f}")
            col2.metric("Mediana (h)", f"{intervals.median():.2f}")
            col3.metric("Mínimo (h)", f"{intervals.min():.2f}")
            col4.metric("Máximo (h)", f"{intervals.max():.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("🚨 Analisador de Alertas - Versão Refatorada")
    st.markdown("### 2 modos: Individual e Completa + CSV (com critérios validados)")
    st.sidebar.header("⚙️ Configurações")
    analysis_mode = st.sidebar.selectbox("🎯 Modo de Análise", ["🔍 Individual", "📊 Completa + CSV"])
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])

    if uploaded_file:
        analyzer = StreamlitAlertAnalyzer()
        if analyzer.load_data(uploaded_file):
            if analysis_mode == "🔍 Individual":
                id_counts = analyzer.df_original['u_alert_id'].value_counts()
                id_options = [f"{uid} ({count})" for uid, count in id_counts.items()]
                selected = st.sidebar.selectbox("Short CI", id_options)
                selected_id = selected.split(" (")[0]
                if st.sidebar.button("🚀 Analisar", type="primary"):
                    if analyzer.prepare_individual_analysis(selected_id):
                        st.success(f"Analisando: {selected_id}")
                        tab1, tab2 = st.tabs(["📊 Básico", "🔄 Reincidência"])
                        with tab1:
                            analyzer.show_basic_stats()
                        with tab2:
                            recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
                            recurrence_analyzer.analyze()

            elif analysis_mode == "📊 Completa + CSV":
                st.subheader("📊 Análise Completa COM CRITÉRIOS VALIDADOS")
                if st.sidebar.button("🚀 Executar", type="primary"):
                    st.info("⏱️ Processando com multiprocessing...")
                    progress_bar = st.progress(0)
                    df_consolidated = analyzer.complete_analysis_all_u_alert_id(progress_bar)
                    progress_bar.empty()
                    if df_consolidated is not None and len(df_consolidated) > 0:
                        st.success(f"✅ {len(df_consolidated)} alertas processados!")
                        st.header("📊 Resumo")
                        col1, col2, col3, col4 = st.columns(4)
                        critical = len(df_consolidated[df_consolidated['classification'].str.contains('CRÍTICO', na=False)])
                        col1.metric("🔴 P1", critical)
                        high = len(df_consolidated[df_consolidated['classification'].str.contains('PARCIALMENTE', na=False)])
                        col2.metric("🟠 P2", high)
                        medium = len(df_consolidated[df_consolidated['classification'].str.contains('DETECTÁVEL', na=False)])
                        col3.metric("🟡 P3", medium)
                        low = len(df_consolidated[df_consolidated['classification'].str.contains('NÃO', na=False)])
                        col4.metric("🟢 P4", low)
                        st.subheader("Dataframe Completo")
                        st.dataframe(df_consolidated, use_container_width=True)
                        st.markdown("---")
                        st.subheader("📥 Exportar")
                        col1, col2 = st.columns(2)
                        csv_full = df_consolidated.to_csv(index=False)
                        col1.download_button("⬇️ CSV Completo", csv_full, f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
                        summary_cols = ['u_alert_id', 'score', 'classification', 'total_occurrences']
                        available_summary = [col for col in summary_cols if col in df_consolidated.columns]
                        summary = df_consolidated[available_summary].copy()
                        csv_summary = summary.to_csv(index=False)
                        col2.download_button("⬇️ CSV Resumido", csv_summary, f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
    else:
        st.info("👆 Faça upload de um CSV")
        with st.expander("📖 Instruções e Validação dos Critérios"):
            st.markdown("""
            ### ✅ CRITÉRIOS VALIDADOS

            1. **Regularidade (25%)** - Consistência via CV
            2. **Periodicidade (25%)** - Detecta ciclos via FFT
            3. **Previsibilidade (15%)** - Indica se podemos prever
            4. **Concentração Temporal (20%)** - Horários/dias fixos
            5. **Frequência Absoluta (15%)** - Volume mínimo necessário
            """)


if __name__ == "__main__":
    main()
