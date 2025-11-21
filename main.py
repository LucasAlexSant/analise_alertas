import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import Counter
import math
import os
import json
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CACHE MANAGER - Gerenciamento de Cache
# ============================================================
class CacheManager:
    """Gerencia cache de resultados de análise para evitar reprocessamento."""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.analysis_cache_path = self.cache_dir / "analysis_results.pkl"
        self.metadata_path = self.cache_dir / "metadata.json"
    
    def save_analysis_results(self, df_results, metadata=None):
        """Salva resultados da análise completa em cache."""
        try:
            df_results.to_pickle(self.analysis_cache_path)
            if metadata is None:
                metadata = {}
            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'total_alerts': len(df_results),
                'file_size': os.path.getsize(self.analysis_cache_path)
            })
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            print(f"Erro ao salvar cache: {e}")
            return False
    
    def load_analysis_results(self):
        """Carrega resultados da análise do cache."""
        try:
            if not self.analysis_cache_path.exists():
                return None, None
            df_results = pd.read_pickle(self.analysis_cache_path)
            metadata = {}
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            return df_results, metadata
        except Exception as e:
            print(f"Erro ao carregar cache: {e}")
            return None, None
    
    def has_cache(self):
        """Verifica se existe cache disponível."""
        return self.analysis_cache_path.exists() and self.metadata_path.exists()
    
    def get_cache_info(self):
        """Retorna informações sobre o cache existente."""
        if not self.has_cache():
            return None
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['file_exists'] = self.analysis_cache_path.exists()
            metadata['file_size_mb'] = os.path.getsize(self.analysis_cache_path) / (1024 * 1024)
            return metadata
        except Exception as e:
            print(f"Erro ao obter info do cache: {e}")
            return None
    
    def clear_cache(self):
        """Limpa o cache existente."""
        try:
            if self.analysis_cache_path.exists():
                os.remove(self.analysis_cache_path)
            if self.metadata_path.exists():
                os.remove(self.metadata_path)
            return True
        except Exception as e:
            print(f"Erro ao limpar cache: {e}")
            return False
    
    def save_comparison_results(self, comparison_data, filename="comparison_results.pkl"):
        """Salva resultados de comparação em cache."""
        try:
            cache_path = self.cache_dir / filename
            if isinstance(comparison_data, pd.DataFrame):
                comparison_data.to_pickle(cache_path)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(comparison_data, f)
            return True
        except Exception as e:
            print(f"Erro ao salvar comparação: {e}")
            return False
    
    def load_comparison_results(self, filename="comparison_results.pkl"):
        """Carrega resultados de comparação do cache."""
        try:
            cache_path = self.cache_dir / filename
            if not cache_path.exists():
                return None
            try:
                return pd.read_pickle(cache_path)
            except:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Erro ao carregar comparação: {e}")
            return None


# ============================================================
# CLUSTERING ANALYZER - Análise de Clusters para Comparação
# ============================================================
class ClusteringAnalyzer:
    """Analisa clusters nos dados de reincidência para identificar padrões."""
    
    def __init__(self, df_comparison):
        self.df = df_comparison.copy()
        self.scaler = StandardScaler()
        self.optimal_k = None
        self.clusters = None
        self.features_normalized = None
        self.feature_cols = None
        self.valid_indices = None
    
    def prepare_features(self):
        """Prepara features para clustering."""
        # Features numéricas para clustering
        feature_cols = []
        
        # Features do código
        if 'score' in self.df.columns:
            feature_cols.append('score')
        if 'total_occurrences' in self.df.columns:
            feature_cols.append('total_occurrences')
        if 'total_clears' in self.df.columns:
            feature_cols.append('total_clears')
        if 'clear_percentage' in self.df.columns:
            feature_cols.append('clear_percentage')
        
        # Features do Athena
        if 'reincidence_count' in self.df.columns:
            feature_cols.append('reincidence_count')
        if 'total_athena_records' in self.df.columns:
            feature_cols.append('total_athena_records')
        
        # Adicionar features binárias
        self.df['is_reincident_code_num'] = self.df['is_reincident_code'].astype(int)
        self.df['is_reincident_athena_num'] = self.df['is_reincident_athena'].astype(int)
        feature_cols.extend(['is_reincident_code_num', 'is_reincident_athena_num'])
        
        # Remover NaN
        df_clean = self.df[feature_cols].fillna(0)  # Preencher NaN com 0
        
        # Normalizar features
        features_normalized = self.scaler.fit_transform(df_clean)
        
        self.features_normalized = features_normalized
        self.feature_cols = feature_cols
        self.valid_indices = df_clean.index
        
        return features_normalized, feature_cols, df_clean.index
    
    def find_optimal_clusters(self, max_k=8):
        """Encontra número ideal de clusters usando Silhouette Score."""
        if len(self.features_normalized) < 4:
            return 2, [], []  # Mínimo de clusters se poucos dados
        
        # Limitar k ao número de amostras
        max_k = min(max_k, len(self.features_normalized) - 1)
        k_range = range(2, max_k + 1)
        
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_normalized)
            
            # Silhouette Score
            sil_score = silhouette_score(self.features_normalized, labels)
            silhouette_scores.append(sil_score)
            
            # Inertia (para método Elbow)
            inertias.append(kmeans.inertia_)
        
        # Encontrar o melhor k pelo Silhouette Score
        best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
        self.optimal_k = best_k
        
        return best_k, list(k_range), silhouette_scores, inertias
    
    def perform_clustering(self, n_clusters=None):
        """Executa clustering com número específico de clusters."""
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features_normalized)
        
        self.clusters = labels
        
        # Calcular centróides
        centroids = kmeans.cluster_centers_
        
        # Calcular métricas do cluster
        cluster_metrics = {}
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)
            
            cluster_metrics[i] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(labels) * 100),
                'centroid': centroids[i]
            }
        
        return labels, cluster_metrics, centroids
    
    def analyze_cluster_characteristics(self):
        """Analisa características de cada cluster."""
        df_clustered = self.df.copy()
        df_clustered['cluster'] = self.clusters
        
        cluster_chars = {}
        
        for cluster_id in np.unique(self.clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            characteristics = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'percentage': float(len(cluster_data) / len(df_clustered) * 100),
                
                # Características de reincidência
                'reincident_code_rate': float((cluster_data['is_reincident_code'].sum() / len(cluster_data) * 100) if len(cluster_data) > 0 else 0),
                'reincident_athena_rate': float((cluster_data['is_reincident_athena'].sum() / len(cluster_data) * 100) if len(cluster_data) > 0 else 0),
                
                # Concordância
                'concordance_rate': float(len(cluster_data[cluster_data['status_comparacao'].str.contains('CONCORDAM', na=False)]) / len(cluster_data) * 100 if len(cluster_data) > 0 else 0),
                
                # Médias
                'avg_score': float(cluster_data['score'].mean()) if 'score' in cluster_data.columns else 0,
                'avg_occurrences': float(cluster_data['total_occurrences'].mean()) if 'total_occurrences' in cluster_data.columns else 0,
                'avg_clear_pct': float(cluster_data['clear_percentage'].mean()) if 'clear_percentage' in cluster_data.columns else 0,
                
                # Status predominante
                'predominant_status': cluster_data['status_comparacao'].mode().values[0] if len(cluster_data) > 0 and len(cluster_data['status_comparacao'].mode()) > 0 else 'N/A',
                
                # Exemplos de alertas
                'sample_alerts': cluster_data['u_alert_id'].head(5).tolist() if 'u_alert_id' in cluster_data.columns else []
            }
            
            # Identificar tipo de cluster baseado nas características
            cluster_type = self._identify_cluster_type(characteristics)
            characteristics['cluster_type'] = cluster_type
            characteristics['cluster_label'] = self._get_cluster_label(cluster_type, characteristics)
            
            cluster_chars[cluster_id] = characteristics
        
        return cluster_chars
    
    def _identify_cluster_type(self, chars):
        """Identifica o tipo de cluster baseado nas características."""
        # Alta concordância e alta reincidência
        if chars['concordance_rate'] > 80 and chars['reincident_code_rate'] > 70:
            return "HIGH_RISK_AGREED"
        
        # Alta concordância e baixa reincidência
        elif chars['concordance_rate'] > 80 and chars['reincident_code_rate'] < 30:
            return "LOW_RISK_AGREED"
        
        # Divergência onde código diz sim mas Athena diz não
        elif chars['concordance_rate'] < 50 and chars['reincident_code_rate'] > chars['reincident_athena_rate'] + 20:
            return "CODE_OVERESTIMATE"
        
        # Divergência onde Athena diz sim mas código diz não
        elif chars['concordance_rate'] < 50 and chars['reincident_athena_rate'] > chars['reincident_code_rate'] + 20:
            return "CODE_UNDERESTIMATE"
        
        # Alta taxa de clear
        elif chars['avg_clear_pct'] > 80:
            return "WELL_MANAGED"
        
        # Baixa taxa de clear com alta reincidência
        elif chars['avg_clear_pct'] < 20 and chars['reincident_code_rate'] > 50:
            return "CRITICAL_UNMANAGED"
        
        # Padrão misto
        else:
            return "MIXED_PATTERN"
    
    def _get_cluster_label(self, cluster_type, chars):
        """Retorna um label descritivo para o cluster."""
        labels = {
            "HIGH_RISK_AGREED": f"🔴 Alto Risco Confirmado ({chars['size']} alertas)",
            "LOW_RISK_AGREED": f"🟢 Baixo Risco Confirmado ({chars['size']} alertas)",
            "CODE_OVERESTIMATE": f"⚠️ Código Superestima ({chars['size']} alertas)",
            "CODE_UNDERESTIMATE": f"⚠️ Athena Indica Mais ({chars['size']} alertas)",
            "WELL_MANAGED": f"✅ Bem Gerenciado ({chars['size']} alertas)",
            "CRITICAL_UNMANAGED": f"🚨 Crítico Não Gerenciado ({chars['size']} alertas)",
            "MIXED_PATTERN": f"🔄 Padrão Misto ({chars['size']} alertas)"
        }
        return labels.get(cluster_type, f"Cluster {chars['cluster_id']} ({chars['size']} alertas)")
    
    def create_visualizations(self):
        """Cria visualizações dos clusters."""
        # PCA para redução de dimensionalidade
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features_normalized)
        
        # DataFrame para visualização
        df_viz = pd.DataFrame({
            'PC1': features_pca[:, 0],
            'PC2': features_pca[:, 1],
            'Cluster': self.clusters,
            'Alert_ID': self.df.loc[self.valid_indices, 'u_alert_id'].values,
            'Status': self.df.loc[self.valid_indices, 'status_comparacao'].values
        })
        
        return df_viz, pca
    
    def generate_cluster_report(self):
        """Gera relatório detalhado dos clusters."""
        cluster_chars = self.analyze_cluster_characteristics()
        
        report = {
            'summary': {
                'total_clusters': len(cluster_chars),
                'optimal_k': self.optimal_k,
                'total_alerts': len(self.df),
                'features_used': self.feature_cols
            },
            'clusters': cluster_chars,
            'recommendations': self._generate_recommendations(cluster_chars)
        }
        
        return report
    
    def _generate_recommendations(self, cluster_chars):
        """Gera recomendações baseadas nos clusters identificados."""
        recommendations = []
        
        for cluster_id, chars in cluster_chars.items():
            cluster_type = chars['cluster_type']
            
            if cluster_type == "HIGH_RISK_AGREED":
                recommendations.append({
                    'cluster': cluster_id,
                    'priority': 'ALTA',
                    'action': f"🔴 Cluster {cluster_id}: Requer ação imediata. {chars['size']} alertas confirmados como reincidentes por ambos os sistemas.",
                    'details': f"Score médio: {chars['avg_score']:.1f}, Ocorrências médias: {chars['avg_occurrences']:.1f}"
                })
            
            elif cluster_type == "CRITICAL_UNMANAGED":
                recommendations.append({
                    'cluster': cluster_id,
                    'priority': 'CRÍTICA',
                    'action': f"🚨 Cluster {cluster_id}: Situação crítica! {chars['size']} alertas reincidentes com apenas {chars['avg_clear_pct']:.1f}% de clear.",
                    'details': "Implementar processo de encerramento urgente."
                })
            
            elif cluster_type == "CODE_OVERESTIMATE":
                recommendations.append({
                    'cluster': cluster_id,
                    'priority': 'MÉDIA',
                    'action': f"⚠️ Cluster {cluster_id}: Revisar critérios do código. {chars['size']} alertas podem estar sendo classificados incorretamente como reincidentes.",
                    'details': f"Divergência: Código {chars['reincident_code_rate']:.1f}% vs Athena {chars['reincident_athena_rate']:.1f}%"
                })
            
            elif cluster_type == "CODE_UNDERESTIMATE":
                recommendations.append({
                    'cluster': cluster_id,
                    'priority': 'MÉDIA',
                    'action': f"⚠️ Cluster {cluster_id}: Código pode estar perdendo reincidências. {chars['size']} alertas identificados pelo Athena mas não pelo código.",
                    'details': f"Divergência: Código {chars['reincident_code_rate']:.1f}% vs Athena {chars['reincident_athena_rate']:.1f}%"
                })
            
            elif cluster_type == "WELL_MANAGED":
                recommendations.append({
                    'cluster': cluster_id,
                    'priority': 'BAIXA',
                    'action': f"✅ Cluster {cluster_id}: Bem gerenciado! Manter práticas atuais.",
                    'details': f"Taxa de clear: {chars['avg_clear_pct']:.1f}%"
                })
        
        # Ordenar por prioridade
        priority_order = {'CRÍTICA': 0, 'ALTA': 1, 'MÉDIA': 2, 'BAIXA': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations


# ============================================================
# ALERT COMPARATOR - Comparação Código vs Athena
# ============================================================
class AlertComparator:
    """Compara resultados de análise de reincidência entre o código local e o Athena."""
    
    def __init__(self, df_code_results, df_athena):
        self.df_code = df_code_results.copy()
        self.df_athena = df_athena.copy()
        self.comparison_results = None
    
    def _is_reincident_code(self, classification):
        """Verifica se a classificação do código indica reincidência."""
        if pd.isna(classification):
            return False
        classification_str = str(classification).upper()
        # R1 e R2 são considerados reincidentes
        if 'CRÍTICO' in classification_str or 'R1' in classification_str:
            return True
        if 'PARCIALMENTE REINCIDENTE' in classification_str or 'R2' in classification_str:
            return True
        return False
    
    def _is_reincident_athena(self, u_symptom):
        """Verifica se o Athena classifica como reincidência."""
        if pd.isna(u_symptom):
            return False
        return 'reincidência' in str(u_symptom).lower() or 'reincidencia' in str(u_symptom).lower()
    
    def compare(self):
        """Executa a comparação completa entre os dois datasets."""
        # Preparar dados do código
        cols_to_use = ['u_alert_id', 'classification', 'score', 'total_occurrences']
        
        # Adicionar colunas opcionais se existirem
        if 'total_clears' in self.df_code.columns:
            cols_to_use.append('total_clears')
        if 'clear_percentage' in self.df_code.columns:
            cols_to_use.append('clear_percentage')
        if 'priorities' in self.df_code.columns:
            cols_to_use.append('priorities')
        
        df_code_prep = self.df_code[cols_to_use].copy()
        df_code_prep['is_reincident_code'] = df_code_prep['classification'].apply(self._is_reincident_code)
        
        # Preparar dados do Athena - agrupar por u_alert_id
        df_athena_grouped = self.df_athena.groupby('u_alert_id').agg({
            'u_symptom': lambda x: list(x)
        }).reset_index()
        
        df_athena_grouped['symptom_list'] = df_athena_grouped['u_symptom']
        df_athena_grouped['has_reincidence'] = df_athena_grouped['symptom_list'].apply(
            lambda symptoms: any(self._is_reincident_athena(s) for s in symptoms)
        )
        df_athena_grouped['reincidence_count'] = df_athena_grouped['symptom_list'].apply(
            lambda symptoms: sum(1 for s in symptoms if self._is_reincident_athena(s))
        )
        df_athena_grouped['total_athena_records'] = df_athena_grouped['symptom_list'].apply(len)
        
        # Merge dos datasets
        comparison = pd.merge(
            df_code_prep,
            df_athena_grouped[['u_alert_id', 'has_reincidence', 'reincidence_count', 'total_athena_records']],
            on='u_alert_id',
            how='outer',
            indicator=True
        )
        
        comparison.rename(columns={'has_reincidence': 'is_reincident_athena'}, inplace=True)
        
        # Preencher NaN
        comparison['is_reincident_code'] = comparison['is_reincident_code'].fillna(False)
        comparison['is_reincident_athena'] = comparison['is_reincident_athena'].fillna(False)
        comparison['score'] = comparison['score'].fillna(0)
        comparison['total_occurrences'] = comparison['total_occurrences'].fillna(0)
        
        # Criar categorias de comparação
        def categorize_match(row):
            code_r = row['is_reincident_code']
            athena_r = row['is_reincident_athena']
            if code_r and athena_r:
                return '✅ CONCORDAM - Ambos Reincidentes'
            elif not code_r and not athena_r:
                return '✅ CONCORDAM - Ambos Não-Reincidentes'
            elif code_r and not athena_r:
                return '⚠️ DIVERGEM - Código diz SIM, Athena diz NÃO'
            elif not code_r and athena_r:
                return '⚠️ DIVERGEM - Código diz NÃO, Athena diz SIM'
            else:
                return '❓ INDETERMINADO'
        
        comparison['status_comparacao'] = comparison.apply(categorize_match, axis=1)
        
        comparison = comparison.drop('_merge', axis=1)
        
        # Reordenar colunas
        cols_order = [
            'u_alert_id',
            'status_comparacao',
            'is_reincident_code',
            'is_reincident_athena',
            'classification',
            'score',
            'total_occurrences',
            'priorities',
            'total_clears',
            'clear_percentage',
            'reincidence_count',
            'total_athena_records'
        ]
        cols_order = [col for col in cols_order if col in comparison.columns]
        comparison = comparison[cols_order]
        
        self.comparison_results = comparison
        return comparison
    
    def get_summary_statistics(self):
        """Retorna estatísticas resumidas da comparação."""
        if self.comparison_results is None:
            self.compare()
        
        df = self.comparison_results
        total_alerts = len(df)
        
        # Contagens por categoria
        concordam_reincidentes = len(df[df['status_comparacao'] == '✅ CONCORDAM - Ambos Reincidentes'])
        concordam_nao_reincidentes = len(df[df['status_comparacao'] == '✅ CONCORDAM - Ambos Não-Reincidentes'])
        divergem_code_sim = len(df[df['status_comparacao'] == '⚠️ DIVERGEM - Código diz SIM, Athena diz NÃO'])
        divergem_code_nao = len(df[df['status_comparacao'] == '⚠️ DIVERGEM - Código diz NÃO, Athena diz SIM'])
        
        total_concordam = concordam_reincidentes + concordam_nao_reincidentes
        total_divergem = divergem_code_sim + divergem_code_nao
        taxa_concordancia = (total_concordam / total_alerts * 100) if total_alerts > 0 else 0
        
        # Estatísticas de Clear
        clear_stats = {}
        if 'total_clears' in df.columns and 'clear_percentage' in df.columns:
            df_with_clears = df.dropna(subset=['total_clears', 'total_occurrences'])
            if len(df_with_clears) > 0:
                clear_stats = {
                    'total_incidents': int(df_with_clears['total_occurrences'].sum()),
                    'total_clears': int(df_with_clears['total_clears'].sum()),
                    'overall_clear_rate': float(
                        (df_with_clears['total_clears'].sum() / df_with_clears['total_occurrences'].sum() * 100)
                        if df_with_clears['total_occurrences'].sum() > 0 else 0
                    ),
                    'avg_clear_percentage': float(df_with_clears['clear_percentage'].mean()),
                    'alerts_with_100_clear': int((df_with_clears['clear_percentage'] == 100).sum()),
                    'alerts_with_0_clear': int((df_with_clears['clear_percentage'] == 0).sum()),
                    'alerts_partial_clear': int(
                        ((df_with_clears['clear_percentage'] > 0) & (df_with_clears['clear_percentage'] < 100)).sum()
                    )
                }
        
        return {
            'total_alerts': total_alerts,
            'concordam': {
                'total': total_concordam,
                'reincidentes': concordam_reincidentes,
                'nao_reincidentes': concordam_nao_reincidentes,
                'percentual': taxa_concordancia
            },
            'divergem': {
                'total': total_divergem,
                'code_sim_athena_nao': divergem_code_sim,
                'code_nao_athena_sim': divergem_code_nao,
                'percentual': (total_divergem / total_alerts * 100) if total_alerts > 0 else 0
            },
            'metricas_codigo': {
                'total_reincidentes': int(df['is_reincident_code'].sum()),
                'percentual_reincidentes': (df['is_reincident_code'].sum() / total_alerts * 100) if total_alerts > 0 else 0
            },
            'metricas_athena': {
                'total_reincidentes': int(df['is_reincident_athena'].sum()),
                'percentual_reincidentes': (df['is_reincident_athena'].sum() / total_alerts * 100) if total_alerts > 0 else 0
            },
            'clear_stats': clear_stats
        }


# ============================================================
# COMPARAÇÃO DE CSVs COM CLUSTERING AUTOMÁTICO
# ============================================================
def show_comparison_module(cache_manager):
    """Módulo de comparação entre CSV do código e CSV do Athena com clustering automático."""
    st.header("🔄 Comparação: Código vs Athena")
    st.markdown("Compare os resultados de reincidência entre seu código e os dados do Athena com análise automática de clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 CSV do Código (Análise)")
        uploaded_code = st.file_uploader(
            "Upload CSV com resultados da análise",
            type=['csv'],
            key='code_csv',
            help="CSV gerado pela análise completa com colunas: u_alert_id, classification, score, etc."
        )
    
    with col2:
        st.subheader("📥 CSV do Athena")
        uploaded_athena = st.file_uploader(
            "Upload CSV do Athena",
            type=['csv'],
            key='athena_csv',
            help="CSV do Athena com colunas: u_alert_id, u_symptom (contendo 'Reincidência')"
        )
    
    if uploaded_code and uploaded_athena:
        try:
            df_code = pd.read_csv(uploaded_code)
            df_athena = pd.read_csv(uploaded_athena)
            
            st.success(f"✅ CSV Código: {len(df_code)} registros | CSV Athena: {len(df_athena)} registros")
            
            if 'u_alert_id' not in df_code.columns or 'classification' not in df_code.columns:
                st.error("❌ CSV do Código deve conter: 'u_alert_id' e 'classification'")
                return
            
            if 'u_alert_id' not in df_athena.columns or 'u_symptom' not in df_athena.columns:
                st.error("❌ CSV do Athena deve conter: 'u_alert_id' e 'u_symptom'")
                return
            
            if st.button("🚀 Executar Comparação e Clustering", type="primary", use_container_width=True):
                with st.spinner("Comparando dados..."):
                    comparator = AlertComparator(df_code, df_athena)
                    df_comparison = comparator.compare()
                    summary = comparator.get_summary_statistics()
                    
                    cache_manager.save_comparison_results(df_comparison)
                    
                    # ========== RESULTADOS DA COMPARAÇÃO ==========
                    st.markdown("---")
                    st.header("📊 Resultados da Comparação")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📋 Total de Alertas", summary['total_alerts'])
                    col2.metric("✅ Concordância", f"{summary['concordam']['percentual']:.1f}%")
                    col3.metric("⚠️ Divergência", f"{summary['divergem']['percentual']:.1f}%")
                    col4.metric("🔴 Reincidentes (Código)", summary['metricas_codigo']['total_reincidentes'])
                    
                    # ========== ANÁLISE DE CLUSTERING AUTOMÁTICA ==========
                    st.markdown("---")
                    st.header("🎯 Análise de Clustering Automática")
                    
                    with st.spinner("Executando análise de clusters..."):
                        # Inicializar analisador de clustering
                        cluster_analyzer = ClusteringAnalyzer(df_comparison)
                        
                        # Preparar features
                        features, feature_cols, valid_indices = cluster_analyzer.prepare_features()
                        
                        st.info(f"📊 Analisando {len(features)} alertas com {len(feature_cols)} features")
                        
                        # Encontrar número ideal de clusters
                        optimal_k, k_range, silhouette_scores, inertias = cluster_analyzer.find_optimal_clusters()
                        
                        # Mostrar análise do número ideal de clusters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if len(silhouette_scores) > 0:
                                fig_silhouette = go.Figure()
                                fig_silhouette.add_trace(go.Scatter(
                                    x=list(k_range),
                                    y=silhouette_scores,
                                    mode='lines+markers',
                                    marker=dict(size=10),
                                    line=dict(width=2)
                                ))
                                fig_silhouette.add_scatter(
                                    x=[optimal_k],
                                    y=[silhouette_scores[list(k_range).index(optimal_k)]],
                                    mode='markers',
                                    marker=dict(size=15, color='red'),
                                    name='Optimal K'
                                )
                                fig_silhouette.update_layout(
                                    title=f"Silhouette Score (Optimal K={optimal_k})",
                                    xaxis_title="Número de Clusters",
                                    yaxis_title="Silhouette Score",
                                    height=300
                                )
                                st.plotly_chart(fig_silhouette, use_container_width=True)
                        
                        with col2:
                            if len(inertias) > 0:
                                fig_elbow = go.Figure()
                                fig_elbow.add_trace(go.Scatter(
                                    x=list(k_range),
                                    y=inertias,
                                    mode='lines+markers',
                                    marker=dict(size=10),
                                    line=dict(width=2)
                                ))
                                fig_elbow.update_layout(
                                    title="Método Elbow",
                                    xaxis_title="Número de Clusters",
                                    yaxis_title="Inertia",
                                    height=300
                                )
                                st.plotly_chart(fig_elbow, use_container_width=True)
                        
                        st.success(f"✅ Número ideal de clusters identificado: **{optimal_k}**")
                        
                        # Executar clustering com número ideal
                        labels, cluster_metrics, centroids = cluster_analyzer.perform_clustering(optimal_k)
                        
                        # Análise detalhada dos clusters
                        cluster_chars = cluster_analyzer.analyze_cluster_characteristics()
                        
                        # ========== CARACTERÍSTICAS DOS CLUSTERS ==========
                        st.markdown("---")
                        st.subheader("📊 Características dos Clusters")
                        
                        # Criar tabs para cada cluster
                        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(optimal_k)])
                        
                        for i, tab in enumerate(cluster_tabs):
                            with tab:
                                chars = cluster_chars[i]
                                
                                # Header do cluster
                                st.markdown(f"### {chars['cluster_label']}")
                                
                                # Métricas principais
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Tamanho", chars['size'])
                                col2.metric("Percentual", f"{chars['percentage']:.1f}%")
                                col3.metric("Concordância", f"{chars['concordance_rate']:.1f}%")
                                col4.metric("Clear Médio", f"{chars['avg_clear_pct']:.1f}%")
                                
                                # Taxas de reincidência
                                st.markdown("#### 📈 Taxas de Reincidência")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Taxa Código", f"{chars['reincident_code_rate']:.1f}%")
                                col2.metric("Taxa Athena", f"{chars['reincident_athena_rate']:.1f}%")
                                col3.metric("Score Médio", f"{chars['avg_score']:.1f}")
                                
                                # Status predominante
                                st.info(f"**Status Predominante:** {chars['predominant_status']}")
                                
                                # Tipo de cluster
                                cluster_type_labels = {
                                    "HIGH_RISK_AGREED": "🔴 Alto Risco Confirmado",
                                    "LOW_RISK_AGREED": "🟢 Baixo Risco Confirmado",
                                    "CODE_OVERESTIMATE": "⚠️ Código Superestima Risco",
                                    "CODE_UNDERESTIMATE": "⚠️ Athena Indica Mais Risco",
                                    "WELL_MANAGED": "✅ Bem Gerenciado",
                                    "CRITICAL_UNMANAGED": "🚨 Crítico Não Gerenciado",
                                    "MIXED_PATTERN": "🔄 Padrão Misto"
                                }
                                st.warning(f"**Tipo:** {cluster_type_labels.get(chars['cluster_type'], chars['cluster_type'])}")
                                
                                # Exemplos de alertas
                                if chars['sample_alerts']:
                                    st.markdown("#### 📋 Exemplos de Alertas")
                                    st.write(", ".join(chars['sample_alerts'][:5]))
                        
                        # ========== VISUALIZAÇÃO DOS CLUSTERS ==========
                        st.markdown("---")
                        st.subheader("🗺️ Visualização dos Clusters")
                        
                        # PCA visualization
                        df_viz, pca = cluster_analyzer.create_visualizations()
                        
                        # Criar labels descritivos para cada cluster
                        cluster_labels_map = {i: cluster_chars[i]['cluster_label'] for i in range(optimal_k)}
                        df_viz['Cluster_Label'] = df_viz['Cluster'].map(cluster_labels_map)
                        
                        # Scatter plot com PCA
                        fig_pca = px.scatter(
                            df_viz, 
                            x='PC1', 
                            y='PC2', 
                            color='Cluster_Label',
                            hover_data=['Alert_ID', 'Status'],
                            title="Visualização dos Clusters (PCA)",
                            height=500,
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        
                        fig_pca.update_layout(
                            xaxis_title=f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                            yaxis_title=f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        # ========== RECOMENDAÇÕES ==========
                        st.markdown("---")
                        st.subheader("💡 Recomendações Baseadas nos Clusters")
                        
                        report = cluster_analyzer.generate_cluster_report()
                        recommendations = report['recommendations']
                        
                        for rec in recommendations:
                            if rec['priority'] == 'CRÍTICA':
                                st.error(f"**{rec['priority']}:** {rec['action']}")
                            elif rec['priority'] == 'ALTA':
                                st.warning(f"**{rec['priority']}:** {rec['action']}")
                            elif rec['priority'] == 'MÉDIA':
                                st.info(f"**{rec['priority']}:** {rec['action']}")
                            else:
                                st.success(f"**{rec['priority']}:** {rec['action']}")
                            
                            if rec['details']:
                                st.write(f"   📊 {rec['details']}")
                        
                        # ========== DISTRIBUIÇÃO DOS CLUSTERS ==========
                        st.markdown("---")
                        st.subheader("📊 Distribuição dos Clusters")
                        
                        # Gráfico de pizza
                        cluster_sizes = [cluster_chars[i]['size'] for i in range(optimal_k)]
                        cluster_labels = [cluster_chars[i]['cluster_label'] for i in range(optimal_k)]
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=cluster_labels,
                            values=cluster_sizes,
                            hole=0.3
                        )])
                        
                        fig_pie.update_layout(
                            title="Distribuição de Alertas por Cluster",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # ========== ANÁLISE DE CONCORDÂNCIA/DIVERGÊNCIA ==========
                        st.markdown("---")
                        st.subheader("✅ Análise de Concordância")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("✅ Ambos Reincidentes", summary['concordam']['reincidentes'])
                        col2.metric("✅ Ambos Não-Reincidentes", summary['concordam']['nao_reincidentes'])
                        col3.metric("📊 Total Concordam", summary['concordam']['total'])
                        
                        st.markdown("---")
                        st.subheader("⚠️ Análise de Divergência")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("⚠️ Código SIM / Athena NÃO", summary['divergem']['code_sim_athena_nao'])
                        col2.metric("⚠️ Código NÃO / Athena SIM", summary['divergem']['code_nao_athena_sim'])
                        col3.metric("📊 Total Divergem", summary['divergem']['total'])
                        
                        # ========== ESTATÍSTICAS DE CLEAR ==========
                        if summary.get('clear_stats'):
                            st.markdown("---")
                            st.subheader("🔒 Análise de Encerramento (Clear)")
                            clear_stats = summary['clear_stats']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("📊 Total de Incidentes", clear_stats['total_incidents'])
                            col2.metric("✅ Total de Clears", clear_stats['total_clears'])
                            col3.metric("📈 Taxa Geral de Clear", f"{clear_stats['overall_clear_rate']:.1f}%")
                            col4.metric("📊 Média de Clear por Alerta", f"{clear_stats['avg_clear_percentage']:.1f}%")
                        
                        # ========== EXPORTAR RESULTADOS ==========
                        st.markdown("---")
                        st.subheader("📥 Exportar Resultados")
                        
                        # Adicionar coluna de cluster ao dataframe de comparação
                        df_comparison_with_clusters = df_comparison.copy()
                        df_comparison_with_clusters['cluster'] = -1  # Inicializar com -1
                        df_comparison_with_clusters.loc[valid_indices, 'cluster'] = labels
                        
                        # Adicionar labels dos clusters
                        df_comparison_with_clusters['cluster_label'] = df_comparison_with_clusters['cluster'].apply(
                            lambda x: cluster_labels_map.get(x, 'Não Clusterizado') if x != -1 else 'Não Clusterizado'
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # CSV com clusters
                        csv_with_clusters = df_comparison_with_clusters.to_csv(index=False)
                        col1.download_button(
                            "⬇️ CSV com Clusters",
                            csv_with_clusters,
                            f"comparacao_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # Relatório de clusters
                        cluster_report_df = pd.DataFrame(cluster_chars).T
                        csv_cluster_report = cluster_report_df.to_csv()
                        col2.download_button(
                            "⬇️ Relatório de Clusters",
                            csv_cluster_report,
                            f"relatorio_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # Recomendações
                        rec_df = pd.DataFrame(recommendations)
                        csv_recommendations = rec_df.to_csv(index=False)
                        col3.download_button(
                            "⬇️ Recomendações",
                            csv_recommendations,
                            f"recomendacoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # ========== TABELA DE RESULTADOS ==========
                        st.markdown("---")
                        st.subheader("📋 Tabela Completa com Clusters")
                        
                        # Mostrar apenas colunas relevantes
                        display_cols = ['u_alert_id', 'status_comparacao', 'cluster_label', 'score', 
                                      'total_occurrences', 'clear_percentage', 'is_reincident_code', 
                                      'is_reincident_athena']
                        display_cols = [col for col in display_cols if col in df_comparison_with_clusters.columns]
                        
                        st.dataframe(
                            df_comparison_with_clusters[display_cols],
                            use_container_width=True,
                            height=400
                        )
                        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivos: {e}")
            import traceback
            st.error(traceback.format_exc())


# ----------------------------
# Helpers para multiprocessing - COM CORREÇÃO CRÍTICA E PRIORIDADES
# ----------------------------
def analyze_single_u_alert_id_recurrence(u_alert_id, df_original):
    """
    CORREÇÃO CRÍTICA: SEMPRE retorna um dict válido, NUNCA None!
    NOVA FEATURE: Agrega prioridades únicas em array
    """
    try:
        df_ci = df_original[df_original['u_alert_id'] == u_alert_id].copy()
        df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
        df_ci = df_ci.dropna(subset=['created_on']).sort_values('created_on')

        # Calcular clears mesmo para dados insuficientes
        total_clears = 0
        clear_percentage = 0.0
        if 'clear' in df_ci.columns:
            total_clears = int(df_ci['clear'].sum())
            clear_percentage = float((total_clears / len(df_ci) * 100) if len(df_ci) > 0 else 0)
        
        # NOVO: Agregar prioridades únicas
        priorities_list = []
        if 'priority' in df_ci.columns:
            # Pegar todas as prioridades únicas, excluindo NaN
            unique_priorities = df_ci['priority'].dropna().unique().tolist()
            # Converter para string e ordenar
            priorities_list = sorted([str(p) for p in unique_priorities])
        
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
                'predictability_score': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'priorities': priorities_list  # NOVO
            }

        analyzer = AdvancedRecurrenceAnalyzer(df_ci, u_alert_id)
        result = analyzer.analyze_complete_silent()
        
        # GARANTIA: Se analyze_complete_silent retornar None, criar dict padrão
        if result is None:
            return {
                'u_alert_id': u_alert_id,
                'total_occurrences': len(df_ci),
                'score': 0,
                'classification': '⚪ ERRO NA ANÁLISE',
                'mean_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'predictability_score': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'priorities': priorities_list  # NOVO
            }
        
        # Adicionar prioridades ao resultado
        result['priorities'] = priorities_list
        
        return result

    except Exception as e:
        print(f"ERRO ao processar {u_alert_id}: {e}")
        return {
            'u_alert_id': u_alert_id,
            'total_occurrences': 0,
            'score': 0,
            'classification': f'⚪ ERRO: {str(e)[:50]}',
            'mean_interval_hours': None,
            'cv': None,
            'regularity_score': 0,
            'periodicity_detected': False,
            'predictability_score': 0,
            'total_clears': 0,
            'clear_percentage': 0.0,
            'priorities': []  # NOVO
        }


def analyze_chunk_recurrence(u_alert_id_list, df_original):
    """
    CORREÇÃO: Remove filtro - adiciona TODOS os resultados
    """
    results = []
    for u_alert_id in u_alert_id_list:
        result = analyze_single_u_alert_id_recurrence(u_alert_id, df_original)
        # SEMPRE adiciona (result nunca é None agora)
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
        
        # NOVO: Mostrar prioridades se disponível
        if 'priority' in df.columns:
            unique_priorities = df['priority'].dropna().unique()
            if len(unique_priorities) > 0:
                priorities_str = ', '.join(sorted([str(p) for p in unique_priorities]))
                st.info(f"🎯 **Prioridades detectadas:** {priorities_str}")
        
        # Mostrar estatísticas de Clear
        if 'clear' in df.columns:
            total_clears = int(df['clear'].sum())
            clear_percentage = (total_clears / len(df) * 100) if len(df) > 0 else 0
            
            st.markdown("---")
            st.subheader("🔒 Análise de Encerramento (Clear)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Clears", total_clears)
            col2.metric("% Encerrado por Clear", f"{clear_percentage:.1f}%")
            col3.metric("% Sem Clear", f"{100 - clear_percentage:.1f}%")
            
            if clear_percentage == 100:
                st.success("✅ Todos os incidentes foram encerrados por clear!")
            elif clear_percentage == 0:
                st.error("❌ Nenhum incidente foi encerrado por clear")
            else:
                st.info(f"📊 {clear_percentage:.1f}% dos incidentes foram encerrados por clear e {100 - clear_percentage:.1f}% não foram")
        
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
        """
        CORREÇÃO CRÍTICA: SEMPRE retorna dict, NUNCA None!
        """
        df = self._prepare_data()
        
        # NOVO: Agregar prioridades
        priorities_list = []
        if self.df is not None and 'priority' in self.df.columns:
            unique_priorities = self.df['priority'].dropna().unique().tolist()
            priorities_list = sorted([str(p) for p in unique_priorities])
        
        # CORREÇÃO: Mesmo com dados insuficientes, retorna dict válido
        if df is None or len(df) < 3:
            df_basic = self.df if self.df is not None else None
            total_occ = len(df_basic) if df_basic is not None else 0
            
            total_clears = 0
            clear_percentage = 0.0
            if df_basic is not None and 'clear' in df_basic.columns:
                total_clears = int(df_basic['clear'].sum())
                clear_percentage = float((total_clears / total_occ * 100) if total_occ > 0 else 0)
            
            return {
                'u_alert_id': self.alert_id,
                'total_occurrences': total_occ,
                'score': 0,
                'classification': '⚪ DADOS INSUFICIENTES',
                'mean_interval_hours': None,
                'median_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'dominant_period_hours': None,
                'predictability_score': 0,
                'next_occurrence_prediction_hours': None,
                'hourly_concentration': 0,
                'daily_concentration': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'priorities': priorities_list  # NOVO
            }
        
        intervals_hours = df['time_diff_hours'].dropna().values
        
        # CORREÇÃO: Se não há intervalos suficientes, ainda retorna dict válido
        if len(intervals_hours) < 2:
            total_clears = 0
            clear_percentage = 0.0
            if 'clear' in df.columns:
                total_clears = int(df['clear'].sum())
                clear_percentage = float((total_clears / len(df) * 100) if len(df) > 0 else 0)
            
            return {
                'u_alert_id': self.alert_id,
                'total_occurrences': len(df),
                'score': 0,
                'classification': '⚪ INTERVALOS INSUFICIENTES',
                'mean_interval_hours': None,
                'median_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'dominant_period_hours': None,
                'predictability_score': 0,
                'next_occurrence_prediction_hours': None,
                'hourly_concentration': 0,
                'daily_concentration': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'priorities': priorities_list  # NOVO
            }

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

        # Calcular estatísticas de Clear
        total_clears = 0
        clear_percentage = 0.0
        if 'clear' in df.columns:
            total_clears = int(df['clear'].sum())
            clear_percentage = float((total_clears / len(df) * 100) if len(df) > 0 else 0)

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
            'total_clears': total_clears,
            'clear_percentage': clear_percentage,
            'priorities': priorities_list  # NOVO
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
        mediana = np.median(intervals)
        mad = np.median(np.abs(intervals - mediana))
        cv = mad / mediana if mediana > 0 else float('inf')

        # Classificar a regularidade com base no CV robusto
        if cv < 0.20:
            regularity_score, pattern_type, pattern_color = 100, "🟢 ALTAMENTE REGULAR", "green"
        elif cv < 0.40:
            regularity_score, pattern_type, pattern_color = 80, "🟢 REGULAR", "lightgreen"
        elif cv < 0.70:
            regularity_score, pattern_type, pattern_color = 60, "🟡 SEMI-REGULAR", "yellow"
        elif cv < 1.20:
            regularity_score, pattern_type, pattern_color = 35, "🟠 IRREGULAR", "orange"
        else:
            regularity_score, pattern_type, pattern_color = 15, "🔴 MUITO IRREGULAR", "red"

        if render:
            st.subheader("🎯 2. Regularidade (com CV Robusto)")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Classificação:** {pattern_type}")
                st.write(f"**CV Robusto:** {cv:.3f}")
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
            return {
                'periods': [],
                'has_periodicity': False,
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None
            }

        # Normalizar os intervalos
        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        
        # Padding para FFT
        n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
        intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
        
        # Calcular FFT
        fft_vals = fft(intervals_padded)
        freqs = fftfreq(n_padded, d=1)
        
        # Filtrar frequências positivas
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        fft_mag = np.abs(fft_vals[positive_idx])
        
        # Definir thresholds para periodicidade forte e moderada
        strong_threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
        moderate_threshold = np.mean(fft_mag) + np.std(fft_mag)
        
        # Identificar picos significativos
        strong_peaks_idx = fft_mag > strong_threshold
        moderate_peaks_idx = (fft_mag > moderate_threshold) & (fft_mag <= strong_threshold)
        
        # Inicializar variáveis de periodicidade
        dominant_periods = []
        has_strong_periodicity = False
        has_moderate_periodicity = False
        dominant_period_hours = None
        
        # Verificar periodicidade forte
        if np.any(strong_peaks_idx):
            dominant_freqs = freqs_pos[strong_peaks_idx]
            dominant_periods = (1 / dominant_freqs)
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            if len(dominant_periods) > 0:
                has_strong_periodicity = True
                dominant_period_hours = float(dominant_periods[0] * np.mean(intervals))
        
        # Verificar periodicidade moderada
        if not has_strong_periodicity and np.any(moderate_peaks_idx):
            has_moderate_periodicity = True
        
        # Renderizar gráficos e resultados no Streamlit
        if render:
            st.subheader("🔍 3. Periodicidade (FFT)")
            if has_strong_periodicity:
                st.success("🎯 **Periodicidades Fortes Detectadas:**")
                for period in dominant_periods:
                    est_time = period * np.mean(intervals)
                    time_str = f"{est_time:.1f}h" if est_time < 24 else f"{est_time/24:.1f} dias"
                    st.write(f"• Período: **{period:.1f}** ocorrências (~{time_str})")
            elif has_moderate_periodicity:
                st.info("📊 **Periodicidade Moderada Detectada**")
            else:
                st.info("📊 Nenhuma periodicidade detectada")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=1/freqs_pos[:len(freqs_pos)//4],
                y=fft_mag[:len(freqs_pos)//4],
                mode='lines',
                fill='tozeroy'
            ))
            fig.update_layout(
                title="Espectro de Frequência",
                xaxis_title="Período",
                yaxis_title="Magnitude",
                height=300,
                xaxis_type="log"
            )
            st.plotly_chart(fig, use_container_width=True, key=f'fft_{self.alert_id}')
        
        return {
            'periods': list(map(float, dominant_periods)) if len(dominant_periods) else [],
            'has_periodicity': len(dominant_periods) > 0,
            'has_strong_periodicity': has_strong_periodicity,
            'has_moderate_periodicity': has_moderate_periodicity,
            'dominant_period_hours': dominant_period_hours
        }
    
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
        # 1. Regularidade
        regularity_score = results['regularity']['regularity_score'] * 0.25
        
        # 2. Periodicidade
        if results['periodicity'].get('has_strong_periodicity', False):
            periodicity_score = 100 * 0.25
        elif results['periodicity'].get('has_moderate_periodicity', False):
            periodicity_score = 50 * 0.25
        else:
            periodicity_score = 0 * 0.25
        
        # 3. Previsibilidade
        predictability_score = results['predictability']['predictability_score'] * 0.15
        
        # 4. Concentração Temporal
        hourly_conc = results['temporal']['hourly_concentration']
        daily_conc = results['temporal']['daily_concentration']
        concentration_score = 0
        if hourly_conc > 60 or daily_conc > 60:
            concentration_score = 100 * 0.20
        elif hourly_conc > 40 or daily_conc > 40:
            concentration_score = 60 * 0.20
        elif hourly_conc > 30 or daily_conc > 30:
            concentration_score = 30 * 0.20
        
        # 5. Frequência Absoluta
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
        
        # Calcular score final e classificação
        final_score = (regularity_score + periodicity_score + predictability_score + concentration_score + frequency_score)
        
        if final_score >= 70 and total_occurrences >= 10:
            classification = "R1 (REINCIDENTE CRITICO)"
        elif final_score >= 50 and total_occurrences >= 5:
            classification = "R2 (PARCIALMENTE REINCIDENTE)"
        elif final_score >= 35:
            classification = "R3 (PADRÃO DETECTAVEL)"
        else:
            classification = "R4 (NÃO REINCIDENTE)"
        
        return round(float(final_score), 2), classification

    def _final_classification(self, results, df, intervals):
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL")
        final_score, classification = self._calculate_final_score_validated(results, df, intervals)

        if final_score >= 70:
            level, color, priority,  = "CRÍTICO", "red", "R1"
        elif final_score >= 50:
            level, color, priority,  = "ALTO", "orange", "R2"
        elif final_score >= 35:
            level, color, priority,  = "MÉDIO", "yellow", "R3" 
        else:
            level, color, priority,  = "BAIXO", "green", "R4"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
            st.metric("Score de Reincidência", f"{final_score:.0f}/100")
            st.markdown("#### 📊 Breakdown dos Critérios VALIDADOS")
            total_occurrences = len(df)
            period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
            freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
            regularity_pts = results['regularity']['regularity_score'] * 0.25
            periodicity_pts = 100 * 0.25 if results['periodicity'].get('has_strong_periodicity', False) else 50 * 0.25 if results['periodicity'].get('has_moderate_periodicity', False) else 0
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
                '1. Regularidade (25%)': regularity_pts,
                '2. Periodicidade (25%)': periodicity_pts,
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
        
        # Adicionar info de clear e prioridades se disponível
        if 'clear' in df.columns:
            export_data['total_clears'] = int(df['clear'].sum())
            export_data['clear_percentage'] = float((df['clear'].sum() / len(df) * 100) if len(df) > 0 else 0)
        
        # NOVO: Adicionar prioridades
        if 'priority' in df.columns:
            unique_priorities = df['priority'].dropna().unique()
            export_data['priorities'] = ', '.join(sorted([str(p) for p in unique_priorities]))
        
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
            if 'created_on' not in df_raw.columns or 'u_alert_id' not in df_raw.columns:
                st.error("❌ Colunas obrigatórias: 'created_on' e 'u_alert_id'")
                return False
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on']).sort_values(['u_alert_id', 'created_on']).reset_index(drop=True)
            self.df_original = df_raw
            
            unique_alerts = len(df_raw['u_alert_id'].unique())
            st.sidebar.write(f"**Total de Alertas Únicos:** {unique_alerts}")
            
            if 'clear' in df_raw.columns:
                st.sidebar.success("✅ Coluna 'clear' detectada")
            
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
        """
        CORREÇÃO: Garantir que TODOS os alertas sejam processados
        """
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None

            u_alert_id_list = list(self.df_original['u_alert_id'].unique())
            total_expected = len(u_alert_id_list)
            
            st.info(f"🎯 **Total de alertas a processar: {total_expected}**")
            
            use_mp = total_expected > 20

            if use_mp:
                n_processes = min(cpu_count(), total_expected, 8)
                st.info(f"🚀 Usando {n_processes} processos para {total_expected} alertas")
                chunk_size = max(1, total_expected // n_processes)
                chunks = [u_alert_id_list[i:i + chunk_size] for i in range(0, total_expected, chunk_size)]
                process_func = partial(analyze_chunk_recurrence, df_original=self.df_original)

                try:
                    all_results = []
                    with Pool(processes=n_processes) as pool:
                        for idx, chunk_results in enumerate(pool.imap(process_func, chunks)):
                            all_results.extend(chunk_results)
                            if progress_bar:
                                progress = (len(all_results) / total_expected)
                                progress_bar.progress(progress, text=f"✅ {len(all_results)}/{total_expected}")
                    
                    total_processed = len(all_results)
                    if total_processed != total_expected:
                        st.error(f"⚠️ ATENÇÃO: Esperado {total_expected} alertas, processado {total_processed}!")
                        st.error(f"❌ FALTAM {total_expected - total_processed} alertas!")
                    else:
                        st.success(f"✅ TODOS os {total_processed} alertas foram processados com sucesso!")
                    
                    df_results = pd.DataFrame(all_results)

                    if progress_bar:
                        progress_bar.progress(1.0, text=f"✅ Completa! {total_processed}/{total_expected}")
                    
                    return df_results

                except Exception as e:
                    st.warning(f"⚠️ Erro no multiprocessing: {e}. Usando modo sequencial...")
                    use_mp = False

            if not use_mp:
                st.info("🔄 Processando em modo sequencial...")
                all_results = []
                for idx, u_alert_id in enumerate(u_alert_id_list):
                    if progress_bar:
                        progress_bar.progress((idx + 1) / total_expected, text=f"✅ {idx + 1}/{total_expected}")
                    
                    result = analyze_single_u_alert_id_recurrence(u_alert_id, self.df_original)
                    all_results.append(result)
                
                total_processed = len(all_results)
                if total_processed != total_expected:
                    st.error(f"⚠️ ATENÇÃO: Esperado {total_expected} alertas, processado {total_processed}!")
                else:
                    st.success(f"✅ TODOS os {total_processed} alertas foram processados com sucesso!")
                
                df_results = pd.DataFrame(all_results)
                return df_results

        except Exception as e:
            st.error(f"❌ Erro: {e}")
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
# Inicializar cache manager
# ============================================================
@st.cache_resource
def get_cache_manager():
    return CacheManager()


# ============================================================
# MAIN - COM TODOS OS 3 MODOS + CACHE + CLUSTERING + PRIORIDADES
# ============================================================
def main():
    st.title("🚨 Analisador de Alertas - VERSÃO COMPLETA COM CLUSTERING")
    st.markdown("### ✅ Cache + Comparação + Clustering Automático + Prioridades")
    
    cache_manager = get_cache_manager()
    
    st.sidebar.header("⚙️ Configurações")
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo de Análise", 
        ["🔍 Individual", "📊 Completa + CSV", "🔄 Comparação (Código vs Athena)"]
    )
    
    # Mostrar opções de cache se houver cache disponível e não for modo de comparação
    if cache_manager.has_cache() and analysis_mode != "🔄 Comparação (Código vs Athena)":
        cache_info = cache_manager.get_cache_info()
        if cache_info:
            with st.sidebar.expander("💾 Cache Disponível", expanded=True):
                st.info(f"""**Data:** {cache_info.get('timestamp', 'N/A')}
**Alertas:** {cache_info.get('total_alerts', 'N/A')}
**Tamanho:** {cache_info.get('file_size_mb', 0):.2f} MB""")
                
                col1, col2 = st.columns(2)
                use_cache = col1.button("✅ Usar Cache", type="primary", use_container_width=True)
                clear_cache = col2.button("🗑️ Limpar", use_container_width=True)
                
                if clear_cache:
                    if cache_manager.clear_cache():
                        st.success("Cache limpo!")
                        st.rerun()
                
                if use_cache:
                    df_cached, metadata = cache_manager.load_analysis_results()
                    if df_cached is not None:
                        st.sidebar.success("✅ Dados carregados do cache!")
                        
                        st.header("📊 Resultados do Cache")
                        st.info(f"Carregado de: {metadata.get('timestamp', 'N/A')}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        critical = len(df_cached[df_cached['classification'].str.contains('CRÍTICO', na=False)])
                        col1.metric("🔴 R1", critical)
                        high = len(df_cached[df_cached['classification'].str.contains('PARCIALMENTE', na=False)])
                        col2.metric("🟠 R2", high)
                        medium = len(df_cached[df_cached['classification'].str.contains('DETECTÁVEL', na=False)])
                        col3.metric("🟡 R3", medium)
                        low = len(df_cached[df_cached['classification'].str.contains('NÃO', na=False)])
                        col4.metric("🟢 R4", low)
                        
                        # Mostrar estatísticas de Clear se disponível
                        if 'clear_percentage' in df_cached.columns:
                            st.markdown("---")
                            st.subheader("🔒 Estatísticas de Clear")
                            col1, col2, col3 = st.columns(3)
                            avg_clear = df_cached['clear_percentage'].mean()
                            total_100_clear = (df_cached['clear_percentage'] == 100).sum()
                            total_0_clear = (df_cached['clear_percentage'] == 0).sum()
                            col1.metric("📊 Média de Clear", f"{avg_clear:.1f}%")
                            col2.metric("✅ 100% Clear", total_100_clear)
                            col3.metric("❌ 0% Clear", total_0_clear)
                        
                        st.subheader("Dataframe Completo")
                        st.dataframe(df_cached, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("📥 Exportar")
                        col1, col2 = st.columns(2)
                        
                        csv_full = df_cached.to_csv(index=False)
                        col1.download_button(
                            "⬇️ CSV Completo",
                            csv_full,
                            f"completo_cache_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        summary_cols = ['u_alert_id', 'score', 'classification', 'total_occurrences', 'priorities', 'clear_percentage']
                        available_summary = [col for col in summary_cols if col in df_cached.columns]
                        summary = df_cached[available_summary].copy()
                        csv_summary = summary.to_csv(index=False)
                        col2.download_button(
                            "⬇️ CSV Resumido",
                            csv_summary,
                            f"resumo_cache_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        return
    
    # Modo de comparação com clustering
    if analysis_mode == "🔄 Comparação (Código vs Athena)":
        show_comparison_module(cache_manager)
        return
    
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
                        df_filtered = analyzer.df_original[analyzer.df_original['u_alert_id'] == selected_id]
                        with st.expander("📋 Preview"):
                            st.write(f"**Colunas:** {list(df_filtered.columns)}")
                            st.dataframe(df_filtered)  
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
                    st.info("⏱️ Processando com validação de completude e agregação de prioridades...")
                    progress_bar = st.progress(0)
                    df_consolidated = analyzer.complete_analysis_all_u_alert_id(progress_bar)
                    progress_bar.empty()
                    
                    if df_consolidated is not None and len(df_consolidated) > 0:
                        # Salvar no cache
                        metadata = {
                            'source_file': uploaded_file.name,
                            'analysis_mode': 'Completa + CSV'
                        }
                        cache_manager.save_analysis_results(df_consolidated, metadata)
                        
                        st.success(f"✅ {len(df_consolidated)} alertas processados e salvos no cache!")
                        st.header("📊 Resumo")
                        col1, col2, col3, col4 = st.columns(4)
                        critical = len(df_consolidated[df_consolidated['classification'].str.contains('R1', na=False)])
                        col1.metric("🔴 R1", critical)
                        high = len(df_consolidated[df_consolidated['classification'].str.contains('R2', na=False)])
                        col2.metric("🟠 R2", high)
                        medium = len(df_consolidated[df_consolidated['classification'].str.contains('R3', na=False)])
                        col3.metric("🟡 R3", medium)
                        low = len(df_consolidated[df_consolidated['classification'].str.contains('R4', na=False)])
                        col4.metric("🟢 R4", low)
                        
                        # Mostrar estatísticas de Clear se disponível
                        if 'clear_percentage' in df_consolidated.columns:
                            st.markdown("---")
                            st.subheader("🔒 Estatísticas Gerais de Clear")
                            col1, col2, col3, col4 = st.columns(4)
                            avg_clear = df_consolidated['clear_percentage'].mean()
                            total_100_clear = (df_consolidated['clear_percentage'] == 100).sum()
                            total_0_clear = (df_consolidated['clear_percentage'] == 0).sum()
                            total_partial = ((df_consolidated['clear_percentage'] > 0) & (df_consolidated['clear_percentage'] < 100)).sum()
                            
                            col1.metric("📊 Média de Clear", f"{avg_clear:.1f}%")
                            col2.metric("✅ 100% Clear", total_100_clear)
                            col3.metric("🟡 Clear Parcial", total_partial)
                            col4.metric("❌ 0% Clear", total_0_clear)
                            
                            st.info(f"""
                                        **💡 Insights:** 
                                        - **{avg_clear:.1f}%** dos alertas foram encerrados por clear em média
                                        - **{100 - avg_clear:.1f}%** dos alertas NÃO foram encerrados por clear em média
                                        - **{total_100_clear}** alertas com encerramento perfeito (100% clear)
                                        - **{total_0_clear}** alertas nunca foram encerrados por clear (requerem atenção)
                            """)
                        
                        st.subheader("Dataframe Completo")
                        st.dataframe(df_consolidated, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("📥 Exportar")
                        col1, col2 = st.columns(2)
                        csv_full = df_consolidated.to_csv(index=False)
                        col1.download_button(
                            "⬇️ CSV Completo",
                            csv_full,
                            f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        summary_cols = ['u_alert_id', 'score', 'classification', 'total_occurrences', 'priorities', 'clear_percentage']
                        available_summary = [col for col in summary_cols if col in df_consolidated.columns]
                        summary = df_consolidated[available_summary].copy()
                        csv_summary = summary.to_csv(index=False)
                        col2.download_button(
                            "⬇️ CSV Resumido",
                            csv_summary,
                            f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
    else:
        st.info("👆 Faça upload de um CSV")
        with st.expander("📖 Instruções"):
            st.markdown("""
            ### ✅ CRITÉRIOS VALIDADOS

            1. **Regularidade (25%)** - Consistência via CV
            2. **Periodicidade (25%)** - Detecta ciclos via FFT
            3. **Previsibilidade (15%)** - Indica se podemos prever
            4. **Concentração Temporal (20%)** - Horários/dias fixos
            5. **Frequência Absoluta (15%)** - Volume mínimo necessário
            
            ### 🎯 NOVO: Clustering Automático na Comparação
            
            Quando você executar a comparação entre Código e Athena, o sistema automaticamente:
            - Identificará o número ideal de clusters
            - Agrupará alertas com características similares
            - Fornecerá recomendações específicas para cada grupo
            - Visualizará os clusters em gráficos interativos
            """)


if __name__ == "__main__":
    main()
