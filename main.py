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
# CLUSTERING ANALYZER - Análise de Clusters
# ============================================================
class ClusteringAnalyzer:
    """Analisa e agrupa alertas usando clustering automático."""
    
    def __init__(self, df_comparison):
        self.df = df_comparison.copy()
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.optimal_k = None
        self.cluster_stats = None
        self.features_used = None
    
    def _prepare_features(self):
        """Prepara features numéricas para clustering."""
        # Features numéricas disponíveis
        potential_features = [
            'score',
            'total_occurrences',
            'total_clears',
            'clear_percentage',
            'reincidence_count',
            'total_athena_records'
        ]
        
        # Filtrar features que existem no dataframe
        available_features = [f for f in potential_features if f in self.df.columns]
        
        if len(available_features) < 2:
            st.warning("⚠️ Poucas features numéricas disponíveis para clustering")
            return None
        
        # Criar colunas binárias para concordância/divergência
        self.df['is_concordant'] = self.df['status_comparacao'].str.contains('CONCORDAM', na=False).astype(int)
        self.df['is_reincident_code_num'] = self.df['is_reincident_code'].astype(int)
        self.df['is_reincident_athena_num'] = self.df['is_reincident_athena'].astype(int)
        
        available_features.extend(['is_concordant', 'is_reincident_code_num', 'is_reincident_athena_num'])
        
        # Criar matriz de features
        feature_df = self.df[available_features].copy()
        
        # Preencher NaN com 0
        feature_df = feature_df.fillna(0)
        
        self.features_used = available_features
        return feature_df
    
    def find_optimal_clusters(self, max_k=10):
        """Encontra o número ideal de clusters usando Silhouette Score e Elbow Method."""
        feature_df = self._prepare_features()
        if feature_df is None:
            return None
        
        # Normalizar features
        X = self.scaler.fit_transform(feature_df)
        
        # Limitar max_k ao número de amostras
        max_k = min(max_k, len(X) - 1, 15)
        
        if max_k < 2:
            st.warning("⚠️ Dados insuficientes para clustering (mínimo 3 amostras)")
            return None
        
        # Calcular métricas para diferentes valores de k
        k_range = range(2, max_k + 1)
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            silhouette_scores.append(silhouette_score(X, labels))
            inertias.append(kmeans.inertia_)
        
        # Encontrar k ótimo pelo Silhouette Score (maior valor)
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Encontrar k ótimo pelo Elbow Method
        # Calcular a "curvatura" usando segunda derivada
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 2  # +2 porque perdemos 2 elementos nas diferenças
            optimal_k_elbow = list(k_range)[min(elbow_idx, len(k_range) - 1)]
        else:
            optimal_k_elbow = optimal_k_silhouette
        
        # Usar média ponderada (preferência para Silhouette)
        self.optimal_k = optimal_k_silhouette
        
        return {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_elbow': optimal_k_elbow,
            'recommended_k': self.optimal_k,
            'best_silhouette': max(silhouette_scores)
        }
    
    def perform_clustering(self, n_clusters=None):
        """Executa o clustering com o número especificado de clusters."""
        feature_df = self._prepare_features()
        if feature_df is None:
            return None
        
        if n_clusters is None:
            if self.optimal_k is None:
                optimization = self.find_optimal_clusters()
                if optimization is None:
                    return None
            n_clusters = self.optimal_k
        
        # Normalizar features
        X = self.scaler.fit_transform(feature_df)
        
        # Executar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(X)
        
        # Adicionar labels ao dataframe
        self.df['cluster'] = self.cluster_labels
        
        # Calcular estatísticas de cada cluster
        self._calculate_cluster_stats()
        
        return self.df
    
    def _calculate_cluster_stats(self):
        """Calcula estatísticas detalhadas de cada cluster."""
        if self.cluster_labels is None:
            return None
        
        stats = {}
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            
            cluster_stats = {
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(self.df) * 100,
            }
            
            # Estatísticas de concordância
            concordant = cluster_df['status_comparacao'].str.contains('CONCORDAM', na=False).sum()
            divergent = cluster_df['status_comparacao'].str.contains('DIVERGEM', na=False).sum()
            cluster_stats['concordant_count'] = int(concordant)
            cluster_stats['divergent_count'] = int(divergent)
            cluster_stats['concordance_rate'] = (concordant / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
            
            # Estatísticas de reincidência
            reincident_code = cluster_df['is_reincident_code'].sum()
            reincident_athena = cluster_df['is_reincident_athena'].sum()
            cluster_stats['reincident_code'] = int(reincident_code)
            cluster_stats['reincident_athena'] = int(reincident_athena)
            cluster_stats['reincidence_rate_code'] = (reincident_code / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
            cluster_stats['reincidence_rate_athena'] = (reincident_athena / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
            
            # Estatísticas numéricas
            if 'score' in cluster_df.columns:
                cluster_stats['avg_score'] = float(cluster_df['score'].mean()) if not cluster_df['score'].isna().all() else 0
                cluster_stats['min_score'] = float(cluster_df['score'].min()) if not cluster_df['score'].isna().all() else 0
                cluster_stats['max_score'] = float(cluster_df['score'].max()) if not cluster_df['score'].isna().all() else 0
            
            if 'total_occurrences' in cluster_df.columns:
                cluster_stats['avg_occurrences'] = float(cluster_df['total_occurrences'].mean()) if not cluster_df['total_occurrences'].isna().all() else 0
                cluster_stats['total_occurrences'] = int(cluster_df['total_occurrences'].sum()) if not cluster_df['total_occurrences'].isna().all() else 0
            
            if 'clear_percentage' in cluster_df.columns:
                cluster_stats['avg_clear_percentage'] = float(cluster_df['clear_percentage'].mean()) if not cluster_df['clear_percentage'].isna().all() else 0
            
            if 'total_clears' in cluster_df.columns:
                cluster_stats['total_clears'] = int(cluster_df['total_clears'].sum()) if not cluster_df['total_clears'].isna().all() else 0
            
            # Classificação predominante
            if 'classification' in cluster_df.columns:
                classification_counts = cluster_df['classification'].value_counts()
                if len(classification_counts) > 0:
                    cluster_stats['dominant_classification'] = classification_counts.index[0]
                    cluster_stats['dominant_classification_pct'] = (classification_counts.iloc[0] / len(cluster_df) * 100)
            
            # Gerar nome/descrição do cluster
            cluster_stats['name'] = self._generate_cluster_name(cluster_stats, cluster_id)
            cluster_stats['description'] = self._generate_cluster_description(cluster_stats)
            
            stats[cluster_id] = cluster_stats
        
        self.cluster_stats = stats
        return stats
    
    def _generate_cluster_name(self, stats, cluster_id):
        """Gera um nome descritivo para o cluster."""
        concordance = stats.get('concordance_rate', 0)
        reincidence_code = stats.get('reincidence_rate_code', 0)
        avg_score = stats.get('avg_score', 0)
        
        if concordance >= 80 and reincidence_code >= 70:
            return f"🔴 Cluster {cluster_id}: Críticos Confirmados"
        elif concordance >= 80 and reincidence_code < 30:
            return f"🟢 Cluster {cluster_id}: Estáveis Confirmados"
        elif concordance < 50 and reincidence_code >= 50:
            return f"🟠 Cluster {cluster_id}: Divergentes Críticos"
        elif concordance < 50 and reincidence_code < 50:
            return f"🟡 Cluster {cluster_id}: Divergentes Baixo Risco"
        elif avg_score >= 70:
            return f"🔴 Cluster {cluster_id}: Alto Score"
        elif avg_score >= 40:
            return f"🟠 Cluster {cluster_id}: Score Moderado"
        else:
            return f"🟢 Cluster {cluster_id}: Baixo Score"
    
    def _generate_cluster_description(self, stats):
        """Gera uma descrição textual do cluster."""
        descriptions = []
        
        concordance = stats.get('concordance_rate', 0)
        if concordance >= 80:
            descriptions.append("Alta concordância entre Código e Athena")
        elif concordance >= 50:
            descriptions.append("Concordância moderada")
        else:
            descriptions.append("Alta divergência entre Código e Athena")
        
        reincidence = stats.get('reincidence_rate_code', 0)
        if reincidence >= 70:
            descriptions.append("Maioria são reincidentes")
        elif reincidence >= 30:
            descriptions.append("Mix de reincidentes e não-reincidentes")
        else:
            descriptions.append("Maioria não são reincidentes")
        
        avg_score = stats.get('avg_score', 0)
        if avg_score >= 70:
            descriptions.append(f"Score médio alto ({avg_score:.1f})")
        elif avg_score >= 40:
            descriptions.append(f"Score médio moderado ({avg_score:.1f})")
        else:
            descriptions.append(f"Score médio baixo ({avg_score:.1f})")
        
        clear_pct = stats.get('avg_clear_percentage', None)
        if clear_pct is not None:
            if clear_pct >= 80:
                descriptions.append(f"Alta taxa de clear ({clear_pct:.1f}%)")
            elif clear_pct <= 20:
                descriptions.append(f"Baixa taxa de clear ({clear_pct:.1f}%)")
        
        return " | ".join(descriptions)
    
    def get_cluster_dataframe(self, cluster_id):
        """Retorna o dataframe filtrado para um cluster específico."""
        if self.cluster_labels is None:
            return None
        return self.df[self.df['cluster'] == cluster_id].copy()


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
        cols_to_use = ['u_alert_id', 'classification', 'score', 'total_occurrences']
        
        if 'total_clears' in self.df_code.columns:
            cols_to_use.append('total_clears')
        if 'clear_percentage' in self.df_code.columns:
            cols_to_use.append('clear_percentage')
        if 'priorities' in self.df_code.columns:
            cols_to_use.append('priorities')
        
        df_code_prep = self.df_code[cols_to_use].copy()
        df_code_prep['is_reincident_code'] = df_code_prep['classification'].apply(self._is_reincident_code)
        
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
        
        comparison = pd.merge(
            df_code_prep,
            df_athena_grouped[['u_alert_id', 'has_reincidence', 'reincidence_count', 'total_athena_records']],
            on='u_alert_id',
            how='outer',
            indicator=True
        )
        
        comparison.rename(columns={'has_reincidence': 'is_reincident_athena'}, inplace=True)
        
        comparison['is_reincident_code'] = comparison['is_reincident_code'].fillna(False)
        comparison['is_reincident_athena'] = comparison['is_reincident_athena'].fillna(False)
        
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
        
        cols_order = [
            'u_alert_id', 'status_comparacao', 'is_reincident_code', 'is_reincident_athena',
            'classification', 'score', 'total_occurrences', 'priorities',
            'total_clears', 'clear_percentage', 'reincidence_count', 'total_athena_records'
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
        
        concordam_reincidentes = len(df[df['status_comparacao'] == '✅ CONCORDAM - Ambos Reincidentes'])
        concordam_nao_reincidentes = len(df[df['status_comparacao'] == '✅ CONCORDAM - Ambos Não-Reincidentes'])
        divergem_code_sim = len(df[df['status_comparacao'] == '⚠️ DIVERGEM - Código diz SIM, Athena diz NÃO'])
        divergem_code_nao = len(df[df['status_comparacao'] == '⚠️ DIVERGEM - Código diz NÃO, Athena diz SIM'])
        
        total_concordam = concordam_reincidentes + concordam_nao_reincidentes
        total_divergem = divergem_code_sim + divergem_code_nao
        taxa_concordancia = (total_concordam / total_alerts * 100) if total_alerts > 0 else 0
        
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
    
    def get_divergent_cases(self, limit=None):
        """Retorna casos onde há divergência entre código e Athena."""
        if self.comparison_results is None:
            self.compare()
        
        divergent = self.comparison_results[
            self.comparison_results['status_comparacao'].str.contains('DIVERGEM', na=False)
        ].copy()
        
        if limit:
            divergent = divergent.head(limit)
        
        return divergent


# Inicializar cache manager
@st.cache_resource
def get_cache_manager():
    return CacheManager()


# ----------------------------
# Helpers para multiprocessing
# ----------------------------
def analyze_single_u_alert_id_recurrence(u_alert_id, df_original):
    try:
        df_ci = df_original[df_original['u_alert_id'] == u_alert_id].copy()
        df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
        df_ci = df_ci.dropna(subset=['created_on']).sort_values('created_on')

        total_clears = 0
        clear_percentage = 0.0
        if 'clear' in df_ci.columns:
            total_clears = int(df_ci['clear'].sum())
            clear_percentage = float((total_clears / len(df_ci) * 100) if len(df_ci) > 0 else 0)
        
        priorities_list = []
        if 'priority' in df_ci.columns:
            unique_priorities = df_ci['priority'].dropna().unique().tolist()
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
                'priorities': priorities_list
            }

        analyzer = AdvancedRecurrenceAnalyzer(df_ci, u_alert_id)
        result = analyzer.analyze_complete_silent()
        
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
                'priorities': priorities_list
            }
        
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
            'priorities': []
        }


def analyze_chunk_recurrence(u_alert_id_list, df_original):
    results = []
    for u_alert_id in u_alert_id_list:
        result = analyze_single_u_alert_id_recurrence(u_alert_id, df_original)
        results.append(result)
    return results


# ============================================================
# AdvancedRecurrenceAnalyzer
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

    def analyze_complete_silent(self):
        df = self._prepare_data()
        
        priorities_list = []
        if self.df is not None and 'priority' in self.df.columns:
            unique_priorities = self.df['priority'].dropna().unique().tolist()
            priorities_list = sorted([str(p) for p in unique_priorities])
        
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
                'priorities': priorities_list
            }
        
        intervals_hours = df['time_diff_hours'].dropna().values
        
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
                'priorities': priorities_list
            }

        results = {}
        try:
            results['basic_stats'] = self._analyze_basic_statistics(intervals_hours)
        except Exception:
            results['basic_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'cv': 0}

        try:
            results['regularity'] = self._analyze_regularity(intervals_hours)
        except Exception:
            results['regularity'] = {'cv': 0, 'regularity_score': 0}

        try:
            results['periodicity'] = self._analyze_periodicity(intervals_hours)
        except Exception:
            results['periodicity'] = {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}

        try:
            results['predictability'] = self._calculate_predictability(intervals_hours)
        except Exception:
            results['predictability'] = {'predictability_score': 0, 'next_expected_hours': 0}

        try:
            results['temporal'] = self._analyze_temporal_patterns(df)
        except Exception:
            results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}

        total_clears = 0
        clear_percentage = 0.0
        if 'clear' in df.columns:
            total_clears = int(df['clear'].sum())
            clear_percentage = float((total_clears / len(df) * 100) if len(df) > 0 else 0)

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
            'priorities': priorities_list
        }

    def _analyze_basic_statistics(self, intervals):
        return {
            'mean': float(np.mean(intervals)),
            'median': float(np.median(intervals)),
            'std': float(np.std(intervals)),
            'min': float(np.min(intervals)),
            'max': float(np.max(intervals)),
            'cv': float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')),
        }

    def _analyze_regularity(self, intervals):
        mediana = np.median(intervals)
        mad = np.median(np.abs(intervals - mediana))
        cv = mad / mediana if mediana > 0 else float('inf')

        if cv < 0.20:
            regularity_score = 100
        elif cv < 0.40:
            regularity_score = 80
        elif cv < 0.70:
            regularity_score = 60
        elif cv < 1.20:
            regularity_score = 35
        else:
            regularity_score = 15
        
        return {'cv': cv, 'regularity_score': regularity_score}

    def _analyze_periodicity(self, intervals):
        if len(intervals) < 10:
            return {
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None
            }

        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
        intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
        
        fft_vals = fft(intervals_padded)
        freqs = fftfreq(n_padded, d=1)
        
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        fft_mag = np.abs(fft_vals[positive_idx])
        
        strong_threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
        moderate_threshold = np.mean(fft_mag) + np.std(fft_mag)
        
        strong_peaks_idx = fft_mag > strong_threshold
        moderate_peaks_idx = (fft_mag > moderate_threshold) & (fft_mag <= strong_threshold)
        
        dominant_periods = []
        has_strong_periodicity = False
        has_moderate_periodicity = False
        dominant_period_hours = None
        
        if np.any(strong_peaks_idx):
            dominant_freqs = freqs_pos[strong_peaks_idx]
            dominant_periods = (1 / dominant_freqs)
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            if len(dominant_periods) > 0:
                has_strong_periodicity = True
                dominant_period_hours = float(dominant_periods[0] * np.mean(intervals))
        
        if not has_strong_periodicity and np.any(moderate_peaks_idx):
            has_moderate_periodicity = True
        
        return {
            'has_strong_periodicity': has_strong_periodicity,
            'has_moderate_periodicity': has_moderate_periodicity,
            'dominant_period_hours': dominant_period_hours
        }

    def _analyze_temporal_patterns(self, df):
        hourly = df.groupby('hour').size().reindex(range(24), fill_value=0)
        daily = df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
        hourly_conc = float(hourly_pct.nlargest(3).sum()) if len(hourly_pct) > 0 else 0.0
        daily_conc = float(daily_pct.nlargest(3).sum()) if len(daily_pct) > 0 else 0.0
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist() if len(hourly) > 0 else []
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist() if len(daily) > 0 else []

        return {'hourly_concentration': hourly_conc, 'daily_concentration': daily_conc, 'peak_hours': peak_hours, 'peak_days': peak_days}

    def _calculate_predictability(self, intervals):
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

        return {'predictability_score': int(predictability), 'next_expected_hours': mean_interval}

    def _calculate_final_score_validated(self, results, df, intervals):
        regularity_score = results['regularity']['regularity_score'] * 0.25
        
        if results['periodicity'].get('has_strong_periodicity', False):
            periodicity_score = 100 * 0.25
        elif results['periodicity'].get('has_moderate_periodicity', False):
            periodicity_score = 50 * 0.25
        else:
            periodicity_score = 0 * 0.25
        
        predictability_score = results['predictability']['predictability_score'] * 0.15
        
        hourly_conc = results['temporal']['hourly_concentration']
        daily_conc = results['temporal']['daily_concentration']
        concentration_score = 0
        if hourly_conc > 60 or daily_conc > 60:
            concentration_score = 100 * 0.20
        elif hourly_conc > 40 or daily_conc > 40:
            concentration_score = 60 * 0.20
        elif hourly_conc > 30 or daily_conc > 30:
            concentration_score = 30 * 0.20
        
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
            classification = "R1 (REINCIDENTE CRITICO)"
        elif final_score >= 50 and total_occurrences >= 5:
            classification = "R2 (PARCIALMENTE REINCIDENTE)"
        elif final_score >= 35:
            classification = "R3 (PADRÃO DETECTAVEL)"
        else:
            classification = "R4 (NÃO REINCIDENTE)"
        
        return round(float(final_score), 2), classification


# ============================================================
# StreamlitAlertAnalyzer
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
                    
                    df_results = pd.DataFrame(all_results)

                    if progress_bar:
                        progress_bar.progress(1.0, text=f"✅ Completa! {len(all_results)}/{total_expected}")
                    
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


# ============================================================
# VISUALIZAÇÃO DE CLUSTERING
# ============================================================
def show_clustering_results(clustering_analyzer, optimization_results):
    """Exibe os resultados do clustering de forma visual."""
    
    st.markdown("---")
    st.header("🎯 Análise de Clustering")
    
    # Mostrar métricas de otimização
    st.subheader("📊 Determinação do Número Ideal de Clusters")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 K Recomendado", optimization_results['recommended_k'])
    col2.metric("📈 K por Silhouette", optimization_results['optimal_k_silhouette'])
    col3.metric("📉 K por Elbow", optimization_results['optimal_k_elbow'])
    
    # Gráficos de otimização
    col1, col2 = st.columns(2)
    
    with col1:
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(
            x=optimization_results['k_range'],
            y=optimization_results['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10)
        ))
        fig_silhouette.add_vline(
            x=optimization_results['optimal_k_silhouette'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Ótimo: {optimization_results['optimal_k_silhouette']}"
        )
        fig_silhouette.update_layout(
            title="📈 Silhouette Score por Número de Clusters",
            xaxis_title="Número de Clusters (K)",
            yaxis_title="Silhouette Score",
            height=350
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    with col2:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=optimization_results['k_range'],
            y=optimization_results['inertias'],
            mode='lines+markers',
            name='Inércia',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10)
        ))
        fig_elbow.add_vline(
            x=optimization_results['optimal_k_elbow'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Elbow: {optimization_results['optimal_k_elbow']}"
        )
        fig_elbow.update_layout(
            title="📉 Método do Cotovelo (Elbow)",
            xaxis_title="Número de Clusters (K)",
            yaxis_title="Inércia",
            height=350
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.info(f"""
    **💡 Interpretação:**
    - **Silhouette Score**: Quanto maior, melhor a separação entre clusters. Valor ótimo: **{optimization_results['best_silhouette']:.3f}**
    - **Método Elbow**: O "cotovelo" indica onde adicionar mais clusters traz retornos diminuídos
    - **Recomendação**: Usar **{optimization_results['recommended_k']} clusters** baseado no Silhouette Score
    """)


def show_cluster_details(clustering_analyzer):
    """Exibe detalhes de cada cluster."""
    
    st.markdown("---")
    st.subheader("📋 Características de Cada Cluster")
    
    cluster_stats = clustering_analyzer.cluster_stats
    n_clusters = len(cluster_stats)
    
    # Criar tabs para cada cluster
    tab_names = [f"Cluster {i}" for i in range(n_clusters)]
    tabs = st.tabs(tab_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            stats = cluster_stats[i]
            
            # Nome e descrição
            st.markdown(f"### {stats['name']}")
            st.info(f"📝 {stats['description']}")
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Tamanho", f"{stats['size']} ({stats['percentage']:.1f}%)")
            col2.metric("✅ Concordância", f"{stats['concordance_rate']:.1f}%")
            col3.metric("🔴 Reincidência (Código)", f"{stats['reincidence_rate_code']:.1f}%")
            col4.metric("🔵 Reincidência (Athena)", f"{stats['reincidence_rate_athena']:.1f}%")
            
            # Métricas secundárias
            col1, col2, col3, col4 = st.columns(4)
            if 'avg_score' in stats:
                col1.metric("📈 Score Médio", f"{stats['avg_score']:.1f}")
            if 'avg_occurrences' in stats:
                col2.metric("🔢 Média Ocorrências", f"{stats['avg_occurrences']:.1f}")
            if 'avg_clear_percentage' in stats:
                col3.metric("🔒 Taxa Clear Média", f"{stats['avg_clear_percentage']:.1f}%")
            if 'dominant_classification' in stats:
                col4.metric("🏷️ Classificação Dominante", stats['dominant_classification'][:20])
            
            # Mostrar alertas do cluster
            st.markdown("#### 📋 Alertas neste Cluster")
            cluster_df = clustering_analyzer.get_cluster_dataframe(i)
            
            # Selecionar colunas relevantes para exibição
            display_cols = ['u_alert_id', 'status_comparacao', 'score', 'total_occurrences', 
                           'classification', 'clear_percentage']
            display_cols = [c for c in display_cols if c in cluster_df.columns]
            
            st.dataframe(cluster_df[display_cols], use_container_width=True, height=300)


def show_cluster_visualizations(clustering_analyzer, df_comparison):
    """Exibe visualizações avançadas dos clusters."""
    
    st.markdown("---")
    st.subheader("📊 Visualizações dos Clusters")
    
    df = df_comparison.copy()
    
    # Gráfico de dispersão Score vs Ocorrências
    if 'score' in df.columns and 'total_occurrences' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                df,
                x='score',
                y='total_occurrences',
                color='cluster',
                color_continuous_scale='viridis',
                hover_data=['u_alert_id', 'status_comparacao'],
                title="📊 Score vs Ocorrências por Cluster"
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Distribuição de tamanho dos clusters
            cluster_sizes = df['cluster'].value_counts().sort_index()
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=[f"Cluster {i}" for i in cluster_sizes.index],
                    y=cluster_sizes.values,
                    marker_color=px.colors.qualitative.Set2[:len(cluster_sizes)],
                    text=cluster_sizes.values,
                    textposition='auto'
                )
            ])
            fig_bar.update_layout(
                title="📊 Distribuição de Alertas por Cluster",
                xaxis_title="Cluster",
                yaxis_title="Quantidade de Alertas",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gráfico de radar para comparar clusters
    st.markdown("#### 🎯 Comparação Radar dos Clusters")
    
    cluster_stats = clustering_analyzer.cluster_stats
    
    categories = ['Concordância', 'Reincidência Código', 'Reincidência Athena', 'Score Médio', 'Clear %']
    
    fig_radar = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for cluster_id, stats in cluster_stats.items():
        values = [
            stats.get('concordance_rate', 0),
            stats.get('reincidence_rate_code', 0),
            stats.get('reincidence_rate_athena', 0),
            stats.get('avg_score', 0),
            stats.get('avg_clear_percentage', 0)
        ]
        values.append(values[0])  # Fechar o radar
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=f"Cluster {cluster_id}",
            line_color=colors[cluster_id % len(colors)]
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Perfil Comparativo dos Clusters",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Heatmap de características
    st.markdown("#### 🔥 Heatmap de Características dos Clusters")
    
    heatmap_data = []
    for cluster_id, stats in cluster_stats.items():
        heatmap_data.append([
            stats.get('concordance_rate', 0),
            stats.get('reincidence_rate_code', 0),
            stats.get('reincidence_rate_athena', 0),
            stats.get('avg_score', 0),
            stats.get('avg_clear_percentage', 0),
            stats.get('percentage', 0)
        ])
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Concordância %', 'Reincid. Código %', 'Reincid. Athena %', 'Score Médio', 'Clear %', 'Tamanho %'],
        y=[f"Cluster {i}" for i in range(len(heatmap_data))],
        colorscale='RdYlGn',
        text=np.round(heatmap_data, 1),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='Cluster %{y}<br>%{x}: %{z:.1f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title="Métricas por Cluster",
        height=300 + len(cluster_stats) * 30
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


def show_cluster_summary_table(clustering_analyzer):
    """Exibe tabela resumida dos clusters."""
    
    st.markdown("---")
    st.subheader("📋 Resumo Comparativo dos Clusters")
    
    cluster_stats = clustering_analyzer.cluster_stats
    
    summary_data = []
    for cluster_id, stats in cluster_stats.items():
        summary_data.append({
            'Cluster': f"Cluster {cluster_id}",
            'Nome': stats['name'],
            'Tamanho': stats['size'],
            '% do Total': f"{stats['percentage']:.1f}%",
            'Concordância': f"{stats['concordance_rate']:.1f}%",
            'Reincid. Código': f"{stats['reincidence_rate_code']:.1f}%",
            'Reincid. Athena': f"{stats['reincidence_rate_athena']:.1f}%",
            'Score Médio': f"{stats.get('avg_score', 0):.1f}",
            'Clear %': f"{stats.get('avg_clear_percentage', 0):.1f}%"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Exportar
    csv_clusters = df_summary.to_csv(index=False)
    st.download_button(
        "⬇️ Exportar Resumo de Clusters",
        csv_clusters,
        f"clusters_resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# COMPARAÇÃO DE CSVs COM CLUSTERING
# ============================================================
def show_comparison_module(cache_manager):
    """Módulo de comparação entre CSV do código e CSV do Athena COM CLUSTERING."""
    st.header("🔄 Comparação: Código vs Athena + Clustering")
    st.markdown("Compare os resultados de reincidência e agrupe automaticamente os alertas")
    
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
            
            if st.button("🚀 Executar Comparação + Clustering", type="primary", use_container_width=True):
                with st.spinner("Comparando dados e executando clustering..."):
                    # COMPARAÇÃO
                    comparator = AlertComparator(df_code, df_athena)
                    df_comparison = comparator.compare()
                    summary = comparator.get_summary_statistics()
                    
                    # Salvar comparação no cache
                    cache_manager.save_comparison_results(df_comparison)
                    
                    # MOSTRAR RESULTADOS DA COMPARAÇÃO
                    st.markdown("---")
                    st.header("📊 Resultados da Comparação")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📋 Total de Alertas", summary['total_alerts'])
                    col2.metric("✅ Concordância", f"{summary['concordam']['percentual']:.1f}%")
                    col3.metric("⚠️ Divergência", f"{summary['divergem']['percentual']:.1f}%")
                    col4.metric("🔴 Reincidentes (Código)", summary['metricas_codigo']['total_reincidentes'])
                    
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
                    
                    # SEÇÃO DE CLEAR STATS
                    if summary.get('clear_stats'):
                        st.markdown("---")
                        st.subheader("🔒 Análise de Encerramento (Clear)")
                        clear_stats = summary['clear_stats']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📊 Total de Incidentes", clear_stats['total_incidents'])
                        col2.metric("✅ Total de Clears", clear_stats['total_clears'])
                        col3.metric("📈 Taxa Geral de Clear", f"{clear_stats['overall_clear_rate']:.1f}%")
                        col4.metric("📊 Média de Clear por Alerta", f"{clear_stats['avg_clear_percentage']:.1f}%")
                    
                    # ========================================
                    # CLUSTERING AUTOMÁTICO
                    # ========================================
                    st.markdown("---")
                    st.header("🎯 CLUSTERING AUTOMÁTICO")
                    
                    if len(df_comparison) < 5:
                        st.warning("⚠️ Dados insuficientes para clustering (mínimo 5 alertas)")
                    else:
                        with st.spinner("Determinando número ideal de clusters..."):
                            clustering_analyzer = ClusteringAnalyzer(df_comparison)
                            
                            # Encontrar número ideal de clusters
                            optimization_results = clustering_analyzer.find_optimal_clusters(max_k=min(10, len(df_comparison) - 1))
                            
                            if optimization_results:
                                # Mostrar resultados da otimização
                                show_clustering_results(clustering_analyzer, optimization_results)
                                
                                # Executar clustering com K ótimo
                                df_clustered = clustering_analyzer.perform_clustering()
                                
                                if df_clustered is not None:
                                    # Mostrar detalhes de cada cluster
                                    show_cluster_details(clustering_analyzer)
                                    
                                    # Mostrar visualizações
                                    show_cluster_visualizations(clustering_analyzer, df_clustered)
                                    
                                    # Tabela resumo
                                    show_cluster_summary_table(clustering_analyzer)
                                    
                                    # Exportar dados com clusters
                                    st.markdown("---")
                                    st.subheader("📥 Exportar Dados Completos")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    csv_complete = df_clustered.to_csv(index=False)
                                    col1.download_button(
                                        "⬇️ Exportar CSV com Clusters",
                                        csv_complete,
                                        f"comparacao_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                    
                                    # Exportar apenas alertas divergentes
                                    divergentes = df_clustered[df_clustered['status_comparacao'].str.contains('DIVERGEM', na=False)]
                                    csv_divergentes = divergentes.to_csv(index=False)
                                    col2.download_button(
                                        "⬇️ Exportar Apenas Divergentes",
                                        csv_divergentes,
                                        f"divergentes_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                            else:
                                st.error("❌ Não foi possível determinar o número ideal de clusters")
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivos: {e}")
            import traceback
            st.error(traceback.format_exc())


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("🚨 Analisador de Alertas - COM CLUSTERING")
    st.markdown("### ✅ Comparação + Clustering Automático com Número Ideal de K")
    
    cache_manager = get_cache_manager()
    
    st.sidebar.header("⚙️ Configurações")
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo de Análise", 
        ["🔄 Comparação + Clustering", "🔍 Individual", "📊 Completa + CSV"]
    )
    
    # Modo de comparação com clustering (PADRÃO)
    if analysis_mode == "🔄 Comparação + Clustering":
        show_comparison_module(cache_manager)
        return
    
    # Mostrar opções de cache
    if cache_manager.has_cache() and analysis_mode != "🔄 Comparação + Clustering":
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
                        
                        st.subheader("Dataframe Completo")
                        st.dataframe(df_cached, use_container_width=True)
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
                        st.success(f"Analisando: {selected_id}")
                        analyzer.show_basic_stats()

            elif analysis_mode == "📊 Completa + CSV":
                st.subheader("📊 Análise Completa")
                if st.sidebar.button("🚀 Executar", type="primary"):
                    st.info("⏱️ Processando...")
                    progress_bar = st.progress(0)
                    df_consolidated = analyzer.complete_analysis_all_u_alert_id(progress_bar)
                    progress_bar.empty()
                    
                    if df_consolidated is not None and len(df_consolidated) > 0:
                        metadata = {'source_file': uploaded_file.name}
                        cache_manager.save_analysis_results(df_consolidated, metadata)
                        
                        st.success(f"✅ {len(df_consolidated)} alertas processados!")
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
                        
                        st.subheader("Dataframe Completo")
                        st.dataframe(df_consolidated, use_container_width=True)
                        
                        csv_full = df_consolidated.to_csv(index=False)
                        st.download_button(
                            "⬇️ CSV Completo",
                            csv_full,
                            f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
    else:
        st.info("👆 Selecione um modo de análise na barra lateral")
        with st.expander("📖 Instruções"):
            st.markdown("""
            ### 🔄 Comparação + Clustering (NOVO!)
            
            1. Faça upload do CSV do **Código** (com análise de reincidência)
            2. Faça upload do CSV do **Athena**
            3. Clique em **Executar Comparação + Clustering**
            
            O sistema irá:
            - ✅ Comparar os resultados de reincidência
            - 🎯 Determinar automaticamente o número ideal de clusters
            - 📊 Agrupar os alertas por características similares
            - 📈 Mostrar visualizações detalhadas de cada cluster
            """)


if __name__ == "__main__":
    main()
