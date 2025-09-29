import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Analisador de Alertas - Completo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitAlertAnalyzer:
    def __init__(self):
        self.df_original = None
        self.df_all_alerts = None
        self.df = None
        self.dates = None
        self.alert_id = None
        
    def load_data(self, uploaded_file):
        """Carrega dados do CSV"""
        try:
            # Ler CSV
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com {len(df_raw)} registros")
            
            # Mostrar informações do dataset
            with st.expander("📋 Informações do Dataset"):
                st.write(f"**Colunas:** {list(df_raw.columns)}")
                st.write(f"**Shape:** {df_raw.shape}")
                st.dataframe(df_raw.head())
            
            # Verificar colunas necessárias
            if 'created_on' not in df_raw.columns or 'u_alert_id' not in df_raw.columns:
                st.error("❌ Colunas 'created_on' e 'u_alert_id' são obrigatórias!")
                return False
            
            # Converter datas
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on'])
            df_raw = df_raw.sort_values(['u_alert_id', 'created_on']).reset_index(drop=True)
            
            self.df_original = df_raw
            
            # Mostrar IDs disponíveis
            unique_ids = df_raw['u_alert_id'].unique()
            st.sidebar.write(f"**IDs disponíveis:** {len(unique_ids)}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def prepare_individual_analysis(self, alert_id):
        """Prepara dados para análise individual"""
        df_filtered = self.df_original[self.df_original['u_alert_id'] == alert_id].copy()
        
        if len(df_filtered) == 0:
            return False
        
        # Extrair informações temporais
        df_filtered['date'] = df_filtered['created_on'].dt.date
        df_filtered['hour'] = df_filtered['created_on'].dt.hour
        df_filtered['day_of_week'] = df_filtered['created_on'].dt.dayofweek
        df_filtered['day_name'] = df_filtered['created_on'].dt.day_name()
        df_filtered['month'] = df_filtered['created_on'].dt.month
        df_filtered['month_name'] = df_filtered['created_on'].dt.month_name()
        df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])
        df_filtered['is_business_hours'] = (df_filtered['hour'] >= 9) & (df_filtered['hour'] <= 17)
        
        # Calcular intervalos
        df_filtered['time_diff_hours'] = df_filtered['created_on'].diff().dt.total_seconds() / 3600
        df_filtered['time_diff_days'] = df_filtered['created_on'].diff().dt.days
        
        self.df = df_filtered
        self.dates = df_filtered['created_on']
        self.alert_id = alert_id
        
        return True

    def prepare_global_analysis(self):
        """Prepara análise global de todos os alertas"""
        st.header("🌍 Análise Global de Todos os Alertas")
        
        # Calcular métricas por alert_id
        alert_metrics = []
        
        progress_bar = st.progress(0)
        unique_ids = self.df_original['u_alert_id'].unique()
        
        for i, alert_id in enumerate(unique_ids):
            progress_bar.progress((i + 1) / len(unique_ids))
            
            df_alert = self.df_original[self.df_original['u_alert_id'] == alert_id].copy()
            
            if len(df_alert) < 2:  # Precisa de pelo menos 2 registros para calcular intervalos
                continue
            
            # Extrair features temporais
            df_alert['hour'] = df_alert['created_on'].dt.hour
            df_alert['day_of_week'] = df_alert['created_on'].dt.dayofweek
            df_alert['is_weekend'] = df_alert['day_of_week'].isin([5, 6])
            df_alert['is_business_hours'] = (df_alert['hour'] >= 9) & (df_alert['hour'] <= 17)
            
            # Calcular intervalos
            df_alert = df_alert.sort_values('created_on')
            intervals_hours = df_alert['created_on'].diff().dt.total_seconds() / 3600
            intervals_hours = intervals_hours.dropna()
            
            if len(intervals_hours) == 0:
                continue
            
            # Período de análise
            period_days = (df_alert['created_on'].max() - df_alert['created_on'].min()).days + 1
            
            # Métricas calculadas
            metrics = {
                'alert_id': alert_id,
                'total_ocorrencias': len(df_alert),
                'periodo_dias': period_days,
                'freq_dia': len(df_alert) / period_days if period_days > 0 else 0,
                'freq_semana': (len(df_alert) / period_days * 7) if period_days > 0 else 0,
                'freq_mes': (len(df_alert) / period_days * 30) if period_days > 0 else 0,
                'intervalo_medio_h': intervals_hours.mean(),
                'intervalo_mediano_h': intervals_hours.median(),
                'intervalo_std_h': intervals_hours.std(),
                'intervalo_min_h': intervals_hours.min(),
                'intervalo_max_h': intervals_hours.max(),
                'hora_pico': df_alert['hour'].mode().iloc[0] if len(df_alert['hour'].mode()) > 0 else 12,
                'pct_fins_semana': df_alert['is_weekend'].mean() * 100,
                'pct_horario_comercial': df_alert['is_business_hours'].mean() * 100,
                'variabilidade_intervalo': intervals_hours.std() / intervals_hours.mean() if intervals_hours.mean() > 0 else 0,
                'primeiro_alerta': df_alert['created_on'].min(),
                'ultimo_alerta': df_alert['created_on'].max()
            }
            
            alert_metrics.append(metrics)
        
        progress_bar.empty()
        
        self.df_all_alerts = pd.DataFrame(alert_metrics)
        
        # Estatísticas globais
        st.subheader("📊 Estatísticas Globais")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔢 Total de Alert IDs", len(unique_ids))
        with col2:
            st.metric("📈 IDs com Dados Suficientes", len(self.df_all_alerts))
        with col3:
            total_alerts = self.df_original.shape[0]
            st.metric("🚨 Total de Alertas", total_alerts)
        with col4:
            period_total = (self.df_original['created_on'].max() - self.df_original['created_on'].min()).days + 1
            st.metric("📅 Período Total (dias)", period_total)
        
        return len(self.df_all_alerts) > 0

    def show_global_overview(self):
        """Visão geral global"""
        st.subheader("📈 Visão Geral dos Alertas")
        
        # Top alertas mais frequentes
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 Top 10 Alertas Mais Frequentes**")
            top_frequent = self.df_all_alerts.nlargest(10, 'total_ocorrencias')[['alert_id', 'total_ocorrencias', 'freq_dia']]
            top_frequent.columns = ['Alert ID', 'Total Ocorrências', 'Frequência/Dia']
            st.dataframe(top_frequent, use_container_width=True)
        
        with col2:
            st.write("**⚡ Top 10 Alertas Mais Rápidos (Menor Intervalo)**")
            top_fast = self.df_all_alerts.nsmallest(10, 'intervalo_medio_h')[['alert_id', 'intervalo_medio_h', 'total_ocorrencias']]
            top_fast.columns = ['Alert ID', 'Intervalo Médio (h)', 'Total Ocorrências']
            st.dataframe(top_fast, use_container_width=True)
        
        # Distribuições
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Histograma de frequências
            fig_freq = px.histogram(
                self.df_all_alerts, 
                x='freq_dia', 
                title="📊 Distribuição de Frequência (alertas/dia)",
                labels={'freq_dia': 'Alertas por Dia', 'count': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        
        with col2:
            # Histograma de intervalos
            fig_int = px.histogram(
                self.df_all_alerts, 
                x='freq_semana', 
                title="📊 Distribuição de Frequência (alertas/semana)",
                labels={'freq_semana': 'Alertas por semana', 'count': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_int, use_container_width=True)

        with col3:
            # Histograma de intervalos
            fig_int = px.histogram(
                self.df_all_alerts, 
                x='freq_mes', 
                title="📊 Distribuição de Frequência (alertas/mes)",
                labels={'freq_mes': 'Alertas por mes', 'count': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_int, use_container_width=True)

        with col4:
            # Histograma de intervalos
            fig_int = px.histogram(
                self.df_all_alerts, 
                x='intervalo_medio_h', 
                title="⏱️ Distribuição de Intervalos Médios",
                labels={'intervalo_medio_h': 'Intervalo Médio (horas)', 'count': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_int, use_container_width=True)

    def perform_clustering_analysis(self):
        """Análise de clustering para agrupar alertas similares"""
        st.subheader("🎯 Agrupamento de Alertas por Perfil de Comportamento")
        
        # Selecionar features para clustering
        features_for_clustering = [
            'freq_dia', 'intervalo_medio_h', 'intervalo_std_h', 
            'hora_pico', 'pct_fins_semana', 'pct_horario_comercial', 'variabilidade_intervalo'
        ]
        
        # Preparar dados
        X = self.df_all_alerts[features_for_clustering].fillna(0)
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número ótimo de clusters
        st.write("**🔍 Determinando Número Ótimo de Clusters...**")
        
        max_clusters = min(10, len(X) - 1)
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Escolher k baseado no silhouette score
        optimal_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Número Ótimo de Clusters", optimal_k)
        with col2:
            st.metric("📊 Silhouette Score", f"{max(silhouette_scores):.3f}")
        
        # Aplicar clustering final
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans_final.fit_predict(X_scaled)
        
        # Adicionar clusters ao dataframe
        self.df_all_alerts['cluster'] = clusters
        
        # Visualização dos clusters
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot 2D
            fig_scatter = px.scatter(
                self.df_all_alerts,
                x='freq_dia',
                y='intervalo_medio_h',
                color='cluster',
                size='total_ocorrencias',
                hover_data=['alert_id'],
                title="🎨 Clusters: Frequência vs Intervalo Médio"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Distribuição por cluster
            cluster_dist = self.df_all_alerts['cluster'].value_counts().sort_index()
            fig_dist = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                title="📊 Distribuição de Alertas por Cluster",
                labels={'x': 'Cluster', 'y': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        return optimal_k

    def show_cluster_profiles(self, n_clusters):
        """Mostra perfis detalhados de cada cluster"""
        st.subheader("👥 Perfis dos Clusters")
        
        # Calcular médias por cluster
        cluster_profiles = self.df_all_alerts.groupby('cluster').agg({
            'total_ocorrencias': ['mean', 'std', 'count'],
            'freq_dia': ['mean', 'std'],
            'intervalo_medio_h': ['mean', 'std'],
            'hora_pico': 'mean',
            'pct_fins_semana': 'mean',
            'pct_horario_comercial': 'mean',
            'variabilidade_intervalo': 'mean'
        }).round(2)
        
        # Criar tabs para cada cluster
        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(n_clusters)])
        
        for i in range(n_clusters):
            with cluster_tabs[i]:
                cluster_data = self.df_all_alerts[self.df_all_alerts['cluster'] == i]
                
                # Métricas do cluster
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Quantidade de Alertas", len(cluster_data))
                with col2:
                    avg_freq = cluster_data['freq_dia'].mean()
                    st.metric("📈 Freq. Média/Dia", f"{avg_freq:.2f}")
                with col3:
                    avg_interval = cluster_data['intervalo_medio_h'].mean()
                    st.metric("⏱️ Intervalo Médio (h)", f"{avg_interval:.2f}")
                with col4:
                    avg_hour = cluster_data['hora_pico'].mean()
                    st.metric("🕐 Hora Pico Média", f"{avg_hour:.0f}:00")
                
                # Características do cluster
                st.write("**🎯 Características do Cluster:**")
                
                weekend_pct = cluster_data['pct_fins_semana'].mean()
                business_pct = cluster_data['pct_horario_comercial'].mean()
                variability = cluster_data['variabilidade_intervalo'].mean()
                
                characteristics = []
                
                if avg_freq > self.df_all_alerts['freq_dia'].median():
                    characteristics.append("🔥 **Alta frequência**")
                else:
                    characteristics.append("🐌 **Baixa frequência**")
                
                if avg_interval < self.df_all_alerts['intervalo_medio_h'].median():
                    characteristics.append("⚡ **Intervalos curtos**")
                else:
                    characteristics.append("⏳ **Intervalos longos**")
                
                if weekend_pct > 30:
                    characteristics.append("🗓️ **Ativo nos fins de semana**")
                
                if business_pct > 70:
                    characteristics.append("🏢 **Predominantemente em horário comercial**")
                elif business_pct < 30:
                    characteristics.append("🌙 **Predominantemente fora do horário comercial**")
                
                if variability > self.df_all_alerts['variabilidade_intervalo'].median():
                    characteristics.append("📊 **Padrão irregular**")
                else:
                    characteristics.append("📈 **Padrão regular**")
                
                for char in characteristics:
                    st.write(f"• {char}")
                
                # Lista de alertas no cluster
                with st.expander(f"📋 Alertas no Cluster {i}"):
                    cluster_alerts = cluster_data[['alert_id', 'total_ocorrencias', 'freq_dia', 'intervalo_medio_h']].copy()
                    cluster_alerts.columns = ['Alert ID', 'Total Ocorrências', 'Freq/Dia', 'Intervalo Médio (h)']
                    st.dataframe(cluster_alerts, use_container_width=True)

    def show_cluster_recommendations(self):
        """Mostra recomendações baseadas nos clusters"""
        st.subheader("💡 Recomendações por Cluster")
        
        for cluster_id in sorted(self.df_all_alerts['cluster'].unique()):
            cluster_data = self.df_all_alerts[self.df_all_alerts['cluster'] == cluster_id]
            
            avg_freq = cluster_data['freq_dia'].mean()
            avg_interval = cluster_data['intervalo_medio_h'].mean()
            weekend_pct = cluster_data['pct_fins_semana'].mean()
            business_pct = cluster_data['pct_horario_comercial'].mean()
            
            with st.expander(f"🎯 Recomendações para Cluster {cluster_id} ({len(cluster_data)} alertas)"):
                recommendations = []
                
                if avg_freq > 5:  # Alta frequência
                    recommendations.append("🚨 **Prioridade Alta**: Alertas muito frequentes - investigar causa raiz")
                    recommendations.append("🔧 **Ação**: Considerar automação de resposta ou ajuste de thresholds")
                
                if avg_interval < 1:  # Intervalos muito curtos
                    recommendations.append("⚡ **Rajadas detectadas**: Possível tempestade de alertas")
                    recommendations.append("🛡️ **Ação**: Implementar rate limiting ou supressão inteligente")
                
                if weekend_pct > 50:
                    recommendations.append("🗓️ **Padrão de fim de semana**: Alertas ativos nos fins de semana")
                    recommendations.append("👥 **Ação**: Verificar cobertura de plantão")
                
                if business_pct < 30:
                    recommendations.append("🌙 **Padrão noturno**: Principalmente fora do horário comercial")
                    recommendations.append("🔄 **Ação**: Considerar processos automatizados noturnos")
                
                if avg_freq < 0.5:  # Baixa frequência
                    recommendations.append("📉 **Baixa frequência**: Alertas esporádicos")
                    recommendations.append("📊 **Ação**: Revisar relevância e configuração do alerta")
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                
                if not recommendations:
                    st.write("• ✅ **Padrão normal**: Nenhuma ação específica recomendada")

    # Manter todos os métodos originais
    def show_basic_stats(self):
        """Exibe estatísticas básicas"""
        st.header("📊 Estatísticas Básicas")
        
        total = len(self.df)
        period_days = (self.dates.max() - self.dates.min()).days + 1
        avg_per_day = total/period_days
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Total de Ocorrências", total)
        with col2:
            st.metric("📅 Período (dias)", period_days)
        with col3:
            st.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        with col4:
            last_alert = self.dates.max().strftime("%d/%m %H:%M")
            st.metric("🕐 Último Alerta", last_alert)
        
        # Intervalos
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            st.subheader("⏱️ Intervalos Entre Alertas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média (h)", f"{intervals.mean():.2f}")
            with col2:
                st.metric("Mediana (h)", f"{intervals.median():.2f}")
            with col3:
                st.metric("Mínimo (h)", f"{intervals.min():.2f}")
            with col4:
                st.metric("Máximo (h)", f"{intervals.max():.2f}")

    def show_temporal_patterns(self):
        """Análise de padrões temporais"""
        st.header("⏰ Padrões Temporais")
        
        # Gráficos side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Por hora
            hourly = self.df['hour'].value_counts().sort_index()
            fig_hour = px.bar(
                x=hourly.index, 
                y=hourly.values,
                title="📊 Distribuição por Hora do Dia",
                labels={'x': 'Hora', 'y': 'Quantidade de Alertas'}
            )
            fig_hour.update_layout(showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)
            
            # Métricas de hora
            peak_hour = hourly.idxmax()
            quiet_hour = hourly.idxmin()
            st.write(f"🕐 **Pico:** {peak_hour:02d}:00 ({hourly[peak_hour]} alertas)")
            st.write(f"🌙 **Menor atividade:** {quiet_hour:02d}:00 ({hourly[quiet_hour]} alertas)")
        
        with col2:
            # Por dia da semana
            daily = self.df['day_name'].value_counts()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_ordered = daily.reindex(days_order).fillna(0)
            
            fig_day = px.bar(
                x=daily_ordered.index, 
                y=daily_ordered.values,
                title="📅 Distribuição por Dia da Semana",
                labels={'x': 'Dia', 'y': 'Quantidade de Alertas'}
            )
            fig_day.update_layout(showlegend=False)
            st.plotly_chart(fig_day, use_container_width=True)
            
            # Métricas de dia
            busiest_day = daily.idxmax()
            st.write(f"📈 **Dia mais ativo:** {busiest_day} ({daily[busiest_day]} alertas)")
        
        # Estatísticas adicionais
        col1, col2 = st.columns(2)
        
        with col1:
            # Business hours
            business = self.df['is_business_hours'].sum()
            non_business = len(self.df) - business
            
            st.subheader("🏢 Horário Comercial (9h-17h)")
            business_data = pd.DataFrame({
                'Período': ['Comercial', 'Fora do horário'],
                'Quantidade': [business, non_business],
                'Porcentagem': [business/len(self.df)*100, non_business/len(self.df)*100]
            })
            
            fig_business = px.pie(
                business_data, 
                values='Quantidade', 
                names='Período',
                title="Distribuição por Horário"
            )
            st.plotly_chart(fig_business, use_container_width=True)
        
        with col2:
            # Weekend
            weekend = self.df['is_weekend'].sum()
            weekday = len(self.df) - weekend
            
            st.subheader("🗓️ Fins de Semana vs Dias Úteis")
            weekend_data = pd.DataFrame({
                'Período': ['Dias úteis', 'Fins de semana'],
                'Quantidade': [weekday, weekend],
                'Porcentagem': [weekday/len(self.df)*100, weekend/len(self.df)*100]
            })
            
            fig_weekend = px.pie(
                weekend_data, 
                values='Quantidade', 
                names='Período',
                title="Distribuição Semanal"
            )
            st.plotly_chart(fig_weekend, use_container_width=True)

    def show_burst_analysis(self):
        """Análise de rajadas"""
        st.header("💥 Análise de Rajadas")
        
        # Slider para threshold
        burst_threshold = st.slider("⏱️ Threshold para Rajada (horas)", 0.5, 24.0, 2.0, 0.5)
        
        intervals = self.df['time_diff_hours'].fillna(999)
        
        bursts = []
        current_burst = []
        
        for i, interval in enumerate(intervals):
            if interval <= burst_threshold and i > 0:
                if not current_burst:
                    current_burst = [i-1, i]
                else:
                    current_burst.append(i)
            else:
                if len(current_burst) >= 2:
                    bursts.append(current_burst)
                current_burst = []
        
        if len(current_burst) >= 2:
            bursts.append(current_burst)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Rajadas Detectadas", len(bursts))
        
        if bursts:
            burst_sizes = [len(b) for b in bursts]
            with col2:
                st.metric("📊 Tamanho Médio", f"{np.mean(burst_sizes):.1f}")
            with col3:
                st.metric("📈 Maior Rajada", f"{max(burst_sizes)} alertas")
            
            # Detalhar maiores rajadas
            st.subheader("🔥 Maiores Rajadas")
            sorted_bursts = sorted(bursts, key=len, reverse=True)[:5]
            
            burst_data = []
            for i, burst_indices in enumerate(sorted_bursts):
                start_time = self.df.iloc[burst_indices[0]]['created_on']
                end_time = self.df.iloc[burst_indices[-1]]['created_on']
                duration = end_time - start_time
                
                burst_data.append({
                    'Rajada': f"#{i+1}",
                    'Alertas': len(burst_indices),
                    'Início': start_time.strftime("%d/%m/%Y %H:%M"),
                    'Fim': end_time.strftime("%d/%m/%Y %H:%M"),
                    'Duração': str(duration)
                })
            
            st.dataframe(pd.DataFrame(burst_data), use_container_width=True)

    def show_trend_analysis(self):
        """Análise de tendências"""
        st.header("📈 Análise de Tendências")
        
        # Agrupar por dia
        daily_counts = self.df.groupby('date').size()
        
        if len(daily_counts) >= 7:
            # Gráfico de linha com tendência
            fig = go.Figure()
            
            # Dados originais
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts.values,
                mode='lines+markers',
                name='Alertas por dia',
                line=dict(color='blue')
            ))
            
            # Linha de tendência
            x_numeric = np.arange(len(daily_counts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_counts.values)
            trend_line = slope * x_numeric + intercept
            
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=trend_line,
                mode='lines',
                name='Tendência',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="📊 Evolução Temporal dos Alertas",
                xaxis_title="Data",
                yaxis_title="Número de Alertas",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas de tendência
            if slope > 0.01:
                trend = "CRESCENTE 📈"
                color = "red"
            elif slope < -0.01:
                trend = "DECRESCENTE 📉"
                color = "green"
            else:
                trend = "ESTÁVEL ➡️"
                color = "blue"
            
            strength = "FORTE" if abs(r_value) > 0.7 else "MODERADA" if abs(r_value) > 0.3 else "FRACA"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Tendência", trend)
            with col2:
                st.metric("💪 Força", strength)
            with col3:
                st.metric("📊 Correlação", f"{r_value:.4f}")
            with col4:
                st.metric("⚡ Taxa/dia", f"{slope:.4f}")
        else:
            st.warning("⚠️ Poucos dados para análise de tendência (mínimo 7 dias)")

    def show_anomaly_detection(self):
        """Detecção de anomalias"""
        st.header("🚨 Detecção de Anomalias")
        
        intervals = self.df['time_diff_hours'].dropna()
        
        if len(intervals) > 4:
            Q1 = intervals.quantile(0.25)
            Q3 = intervals.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            fast_anomalies = intervals[intervals < lower_bound]
            slow_anomalies = intervals[intervals > upper_bound]
            normal_intervals = intervals[(intervals >= lower_bound) & (intervals <= upper_bound)]
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Intervalos Curtos", len(fast_anomalies))
            with col2:
                st.metric("🐌 Intervalos Longos", len(slow_anomalies))
            with col3:
                st.metric("✅ Intervalos Normais", len(normal_intervals))
            
            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=intervals,
                name="Intervalos (horas)",
                boxpoints='outliers'
            ))
            fig.update_layout(
                title="📊 Distribuição dos Intervalos (Detecção de Outliers)",
                yaxis_title="Horas"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detalhes das anomalias
            if len(fast_anomalies) > 0 or len(slow_anomalies) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(fast_anomalies) > 0:
                        st.subheader("⚡ Intervalos Muito Curtos")
                        st.write(f"Menor intervalo: **{fast_anomalies.min():.2f} horas**")
                        st.write(f"Média dos curtos: **{fast_anomalies.mean():.2f} horas**")
                
                with col2:
                    if len(slow_anomalies) > 0:
                        st.subheader("🐌 Intervalos Muito Longos")
                        st.write(f"Maior intervalo: **{slow_anomalies.max():.2f} horas**")
                        st.write(f"Média dos longos: **{slow_anomalies.mean():.2f} horas**")
        else:
            st.warning("⚠️ Poucos dados para detecção de anomalias")

    def show_predictions(self):
        """Insights preditivos"""
        st.header("🔮 Insights Preditivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Horário mais provável
            st.subheader("⏰ Horários de Maior Probabilidade")
            hourly_prob = self.df['hour'].value_counts(normalize=True).sort_values(ascending=False)
            
            prob_data = []
            for hour, prob in hourly_prob.head(5).items():
                prob_data.append({
                    'Horário': f"{hour:02d}:00",
                    'Probabilidade': f"{prob*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True)
        
        with col2:
            # Dia mais provável
            st.subheader("📅 Dias de Maior Probabilidade")
            daily_prob = self.df['day_name'].value_counts(normalize=True).sort_values(ascending=False)
            
            day_data = []
            for day, prob in daily_prob.items():
                day_data.append({
                    'Dia': day,
                    'Probabilidade': f"{prob*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(day_data), use_container_width=True)
        
        # Previsão próximo alerta
        st.subheader("⏱️ Previsão do Próximo Alerta")
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            avg_interval = intervals.mean()
            median_interval = intervals.median()
            last_alert = self.dates.max()
            
            next_avg = last_alert + timedelta(hours=avg_interval)
            next_median = last_alert + timedelta(hours=median_interval)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🕐 Último Alerta", last_alert.strftime("%d/%m %H:%M"))
            with col2:
                st.metric("📊 Próximo (Média)", next_avg.strftime("%d/%m %H:%M"))
            with col3:
                st.metric("📈 Próximo (Mediana)", next_median.strftime("%d/%m %H:%M"))
            
            st.info(f"💡 **Baseado em:** Intervalo médio de {avg_interval:.1f}h e mediana de {median_interval:.1f}h")

def main():
    st.title("🚨 Analisador de Alertas - Versão Completa")
    st.markdown("### Análise individual, global e agrupamento inteligente de alertas")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção do modo de análise
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo de Análise",
        ["🌍 Análise Global", "🔍 Análise Individual"],
        help="Escolha entre analisar todos os alertas ou um alerta específico"
    )
    
    # Upload de arquivo
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload do arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo CSV contendo os dados dos alertas"
    )
    
    if uploaded_file is not None:
        # Inicializar analyzer
        analyzer = StreamlitAlertAnalyzer()
        
        # Carregar dados
        if analyzer.load_data(uploaded_file):
            
            if analysis_mode == "🌍 Análise Global":
                st.markdown("---")
                
                # Botão para análise global
                if st.sidebar.button("🚀 Executar Análise Global", type="primary"):
                    
                    # Preparar análise global
                    if analyzer.prepare_global_analysis():
                        
                        # Tabs para análise global
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📊 Visão Geral",
                            "🎯 Agrupamento", 
                            "👥 Perfis dos Clusters",
                            "💡 Recomendações"
                        ])
                        
                        with tab1:
                            analyzer.show_global_overview()
                        
                        with tab2:
                            n_clusters = analyzer.perform_clustering_analysis()
                        
                        with tab3:
                            analyzer.show_cluster_profiles(n_clusters)
                        
                        with tab4:
                            analyzer.show_cluster_recommendations()
                        
                        # Download dos resultados
                        st.sidebar.markdown("---")
                        st.sidebar.subheader("📥 Downloads")
                        
                        # Download da análise global
                        csv_buffer = io.StringIO()
                        analyzer.df_all_alerts.to_csv(csv_buffer, index=False)
                        st.sidebar.download_button(
                            label="⬇️ Baixar Análise Global",
                            data=csv_buffer.getvalue(),
                            file_name=f"analise_global_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error("❌ Não foi possível processar os dados para análise global")
            
            else:  # Análise Individual
                # Preview dos dados
                try:
                    unique_ids = analyzer.df_original['u_alert_id'].unique()
                    selected_id = st.sidebar.selectbox(
                        "🎯 Selecione o Alert ID",
                        unique_ids,
                        help="Escolha o ID do alerta para análise"
                    )
                    
                    # Botão para análise individual
                    if st.sidebar.button("🚀 Executar Análise Individual", type="primary"):
                        # Preparar e processar dados individuais
                        if analyzer.prepare_individual_analysis(selected_id):
                            
                            st.success(f"🎯 Analisando alert_id: {selected_id} ({len(analyzer.df)} registros)")
                            st.info(f"📅 **Período analisado:** {analyzer.dates.min()} até {analyzer.dates.max()}")
                            
                            # Tabs para organizar as análises individuais
                            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                                "📊 Básico", 
                                "⏰ Temporais", 
                                "💥 Rajadas", 
                                "📈 Tendências", 
                                "🚨 Anomalias", 
                                "🔮 Previsões"
                            ])
                            
                            with tab1:
                                analyzer.show_basic_stats()
                            
                            with tab2:
                                analyzer.show_temporal_patterns()
                            
                            with tab3:
                                analyzer.show_burst_analysis()
                            
                            with tab4:
                                analyzer.show_trend_analysis()
                            
                            with tab5:
                                analyzer.show_anomaly_detection()
                            
                            with tab6:
                                analyzer.show_predictions()
                            
                            # Download dos dados processados individuais
                            st.sidebar.markdown("---")
                            st.sidebar.subheader("📥 Download")
                            
                            csv_buffer = io.StringIO()
                            analyzer.df.to_csv(csv_buffer, index=False)
                            st.sidebar.download_button(
                                label="⬇️ Baixar Dados Processados",
                                data=csv_buffer.getvalue(),
                                file_name=f"analise_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        
                        else:
                            st.error(f"❌ Nenhum registro encontrado para alert_id: {selected_id}")
                            
                except Exception as e:
                    st.error(f"❌ Erro ao processar análise individual: {e}")
    
    else:
        st.info("👆 Faça upload de um arquivo CSV para começar a análise")
        
        # Instruções expandidas
        with st.expander("📖 Instruções de Uso"):
            st.markdown("""
            ### Como usar este analisador:
            
            #### 🌍 **Análise Global**
            1. **📁 Upload do arquivo:** Carregue um arquivo CSV com os dados dos alertas
            2. **🎯 Selecione "Análise Global"** no modo de análise
            3. **🚀 Clique em "Executar Análise Global"**
            4. **📊 Explore os resultados** nas diferentes abas:
               - **Visão Geral:** Estatísticas gerais e rankings
               - **Agrupamento:** Clustering automático de alertas similares
               - **Perfis dos Clusters:** Características detalhadas de cada grupo
               - **Recomendações:** Sugestões de ação para cada cluster
            
            #### 🔍 **Análise Individual**
            1. **📁 Upload do arquivo:** Carregue um arquivo CSV com os dados dos alertas
            2. **🎯 Selecione "Análise Individual"** no modo de análise
            3. **🎯 Escolha um Alert ID** específico
            4. **🚀 Clique em "Executar Análise Individual"**
            5. **📊 Navegue pelas abas** para ver diferentes análises detalhadas
            
            ### Colunas necessárias no CSV:
            - `u_alert_id`: Identificador único do alerta
            - `created_on`: Data e hora da criação do alerta
            
            ### 🎯 **Funcionalidades da Análise Global:**
            - **📊 Ranking de alertas** por frequência e velocidade
            - **🎨 Clustering inteligente** para agrupar alertas com comportamentos similares
            - **👥 Perfis detalhados** de cada cluster identificado
            - **💡 Recomendações específicas** por tipo de comportamento
            - **📈 Visualizações comparativas** entre diferentes alertas
            
            ### 🔍 **Funcionalidades da Análise Individual:**
            - **📊 Básico:** Estatísticas gerais e métricas principais
            - **⏰ Temporais:** Padrões por hora, dia da semana, etc.
            - **💥 Rajadas:** Detecção de sequências rápidas de alertas
            - **📈 Tendências:** Evolução temporal e projeções
            - **🚨 Anomalias:** Identificação de comportamentos atípicos
            - **🔮 Previsões:** Insights preditivos e próximos alertas
            
            ### 🚀 **Principais Melhorias:**
            - **Análise comparativa** de todos os alertas do sistema
            - **Agrupamento automático** usando algoritmos de machine learning
            - **Identificação de padrões** similares entre diferentes alertas
            - **Recomendações inteligentes** baseadas nos clusters encontrados
            - **Interface dual** para análises globais e individuais
            """)

if __name__ == "__main__":
    main()