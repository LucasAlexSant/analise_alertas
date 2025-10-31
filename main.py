import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import io
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import defaultdict, Counter
import math

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas - Completo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNÇÕES DE GERAÇÃO DE REGRAS CUSTOMIZADAS
# ============================================================

def identificar_padrao_dominante(results):
    """
    Identifica o padrão DOMINANTE baseado em todos os scores
    
    Returns:
        tuple: (tipo_padrao, confianca, detalhes)
    """
    padroes_detectados = []
    
    # 1. PERIODICIDADE FORTE (Intervalo fixo: 8 em 8h, 12 em 12h, etc)
    if results['periodicity'].get('has_strong_periodicity', False):
        periodo_horas = results['periodicity'].get('dominant_period_hours')
        
        if periodo_horas:
            confianca = 90 if results['regularity']['regularity_score'] > 80 else 75
            
            # Classificar o tipo de período
            if periodo_horas < 1:
                subtipo = "MINUTOS"
                descricao = f"A cada {periodo_horas * 60:.0f} minutos"
            elif periodo_horas < 12:
                subtipo = "HORARIO_FIXO"
                descricao = f"A cada {periodo_horas:.1f} horas"
            elif 20 <= periodo_horas <= 28:
                subtipo = "DIARIO"
                descricao = "Aproximadamente 1x por dia"
            elif 160 <= periodo_horas <= 180:
                subtipo = "SEMANAL"
                descricao = "Aproximadamente 1x por semana"
            elif 330 <= periodo_horas <= 370:
                subtipo = "QUINZENAL"
                descricao = "Aproximadamente a cada 15 dias"
            else:
                subtipo = "PERIODICO_IRREGULAR"
                descricao = f"Período de {periodo_horas / 24:.1f} dias"
            
            padroes_detectados.append({
                'tipo': 'PERIODICO_FIXO',
                'subtipo': subtipo,
                'confianca': confianca,
                'score_relevancia': 95,
                'periodo_horas': periodo_horas,
                'descricao': descricao
            })
    
    # 2. CONCENTRAÇÃO HORÁRIA (Sempre nos mesmos horários: 8h, 16h, 00h)
    if results['temporal'].get('hourly_concentration', 0) > 60:
        peak_hours = results['temporal'].get('peak_hours', [])
        
        if peak_hours:
            confianca = 85 if len(peak_hours) <= 3 else 70
            
            padroes_detectados.append({
                'tipo': 'HORARIOS_FIXOS',
                'subtipo': 'INTRADIARIO',
                'confianca': confianca,
                'score_relevancia': 90,
                'horarios': peak_hours,
                'concentracao': results['temporal']['hourly_concentration'],
                'descricao': f"Concentrado nos horários: {', '.join([f'{h:02d}:00' for h in peak_hours])}"
            })
    
    # 3. CONCENTRAÇÃO SEMANAL (Sempre nos mesmos dias da semana)
    if results['temporal'].get('daily_concentration', 0) > 60:
        peak_days = results['temporal'].get('peak_days', [])
        
        if peak_days:
            confianca = 80
            dias_map = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            dias_nome = [dias_map[d] for d in peak_days]
            
            padroes_detectados.append({
                'tipo': 'DIAS_FIXOS',
                'subtipo': 'SEMANAL',
                'confianca': confianca,
                'score_relevancia': 85,
                'dias': peak_days,
                'concentracao': results['temporal']['daily_concentration'],
                'descricao': f"Concentrado nos dias: {', '.join(dias_nome)}"
            })
    
    # 4. PADRÃO DE BURST/RAJADA
    if results['bursts'].get('has_bursts', False):
        n_bursts = results['bursts'].get('n_bursts', 0)
        
        if n_bursts >= 2:
            confianca = 75
            
            padroes_detectados.append({
                'tipo': 'BURST',
                'subtipo': 'RAJADAS',
                'confianca': confianca,
                'score_relevancia': 70,
                'num_bursts': n_bursts,
                'descricao': f"Padrão de rajadas ({n_bursts} detectadas)"
            })
    
    # 5. ALTA REGULARIDADE sem periodicidade clara
    if (results['regularity']['regularity_score'] > 70 and 
        not results['periodicity'].get('has_strong_periodicity', False)):
        
        cv = results['regularity']['cv']
        confianca = 65
        
        padroes_detectados.append({
            'tipo': 'REGULAR_SEM_CICLO',
            'subtipo': 'ESTAVEL',
            'confianca': confianca,
            'score_relevancia': 60,
            'cv': cv,
            'descricao': f"Regular mas sem ciclo claro (CV={cv:.2%})"
        })
    
    # 6. MARKOV (Dependência de estado anterior)
    if results['markov'].get('markov_score', 0) > 60:
        markov_score = results['markov']['markov_score']
        confianca = 70
        
        padroes_detectados.append({
            'tipo': 'MARKOV',
            'subtipo': 'DEPENDENTE',
            'confianca': confianca,
            'score_relevancia': 55,
            'markov_score': markov_score,
            'descricao': f"Padrão markoviano (score={markov_score:.1f})"
        })
    
    # 7. COMPORTAMENTO CONTEXTUAL (Fins de semana, feriados, horário comercial)
    if results.get('contextual'):
        weekend_corr = results['contextual'].get('weekend_correlation', 0)
        holiday_corr = results['contextual'].get('holiday_correlation', 0)
        
        if weekend_corr > 0.3:
            confianca = 70
            padroes_detectados.append({
                'tipo': 'CONTEXTUAL',
                'subtipo': 'FINS_DE_SEMANA',
                'confianca': confianca,
                'score_relevancia': 65,
                'correlacao': weekend_corr,
                'descricao': f"Correlacionado com fins de semana ({weekend_corr:.0%})"
            })
        
        if holiday_corr > 0.2:
            confianca = 65
            padroes_detectados.append({
                'tipo': 'CONTEXTUAL',
                'subtipo': 'FERIADOS',
                'confianca': confianca,
                'score_relevancia': 60,
                'correlacao': holiday_corr,
                'descricao': f"Correlacionado com feriados ({holiday_corr:.0%})"
            })
    
    # 8. SEM PADRÃO CLARO (comportamento aleatório)
    if not padroes_detectados or results['randomness'].get('overall_randomness_score', 50) > 70:
        return {
            'tipo': 'SEM_PADRAO_CLARO',
            'subtipo': 'ALEATORIO',
            'confianca': 60,
            'score_relevancia': 0,
            'descricao': 'Comportamento aleatório ou sem padrão detectável'
        }, []
    
    # Ordenar por relevância e confiança
    padroes_detectados.sort(key=lambda x: (x['score_relevancia'], x['confianca']), reverse=True)
    
    # Retornar padrão dominante + padrões secundários
    padrao_principal = padroes_detectados[0] if padroes_detectados else None
    padroes_secundarios = padroes_detectados[1:3] if len(padroes_detectados) > 1 else []
    
    return padrao_principal, padroes_secundarios


def gerar_regra_customizada(results, df, intervals_hours):
    """
    Gera regra customizada baseada no padrão identificado
    
    Returns:
        dict: Regra com parâmetros específicos
    """
    padrao_principal, padroes_secundarios = identificar_padrao_dominante(results)
    
    if not padrao_principal:
        return gerar_regra_generica()
    
    tipo = padrao_principal['tipo']
    
    # REGRAS POR TIPO DE PADRÃO
    
    if tipo == 'PERIODICO_FIXO':
        periodo = padrao_principal['periodo_horas']
        subtipo = padrao_principal['subtipo']
        
        # Calcular tolerância baseada na regularidade
        cv = results['regularity']['cv']
        if cv < 0.2:
            tolerancia_pct = 0.10  # 10% de tolerância para padrões muito regulares
        elif cv < 0.4:
            tolerancia_pct = 0.15
        else:
            tolerancia_pct = 0.25
        
        tolerancia = periodo * tolerancia_pct
        
        # Definir quantas ocorrências consecutivas
        if periodo < 4:  # Menos de 4 horas
            ocorrencias_minimas = 5  # 5 vezes consecutivas
            janela_analise_multiplicador = 6
        elif periodo < 12:
            ocorrencias_minimas = 4
            janela_analise_multiplicador = 5
        elif periodo < 48:
            ocorrencias_minimas = 3
            janela_analise_multiplicador = 4
        else:
            ocorrencias_minimas = 3
            janela_analise_multiplicador = 4
        
        janela_analise_horas = periodo * janela_analise_multiplicador
        
        return {
            'tipo': 'INTERVALO_FIXO',
            'subtipo': subtipo,
            'padrao_principal': padrao_principal,
            'padroes_secundarios': padroes_secundarios,
            'parametros': {
                'periodo_esperado_horas': round(periodo, 2),
                'tolerancia_horas': round(tolerancia, 2),
                'tolerancia_percentual': tolerancia_pct * 100,
                'ocorrencias_consecutivas_minimas': ocorrencias_minimas,
                'janela_analise_horas': round(janela_analise_horas, 2),
                'confianca_padrao': padrao_principal['confianca']
            },
            'criterio_reincidencia': {
                'descricao': f"Considerar REINCIDENTE se ocorrer {ocorrencias_minimas}+ vezes consecutivas",
                'condicao': f"Com intervalo de {periodo:.1f}h (±{tolerancia:.1f}h)",
                'janela': f"Analisando últimas {janela_analise_horas:.0f}h"
            },
            'implementacao': {
                'logica': f"""
                FOR cada novo alerta:
                    ultimos_alertas = buscar_ultimos_alertas(janela={janela_analise_horas:.0f}h)
                    
                    IF len(ultimos_alertas) >= {ocorrencias_minimas}:
                        intervalos = calcular_intervalos(ultimos_alertas)
                        
                        contador_padrao = 0
                        FOR intervalo in intervalos:
                            IF {periodo - tolerancia:.1f} <= intervalo <= {periodo + tolerancia:.1f}:
                                contador_padrao += 1
                        
                        IF contador_padrao >= {ocorrencias_minimas - 1}:
                            MARCAR_COMO_REINCIDENTE()
                            GERAR_ALERTA_CRITICO()
                """
            }
        }
    
    elif tipo == 'HORARIOS_FIXOS':
        horarios = padrao_principal['horarios']
        concentracao = padrao_principal['concentracao']
        
        # Tolerância em minutos
        tolerancia_minutos = 30 if concentracao > 80 else 45
        
        # Quantos dias consecutivos
        dias_consecutivos = 3 if concentracao > 75 else 4
        
        return {
            'tipo': 'HORARIOS_RECORRENTES',
            'subtipo': 'INTRADIARIO',
            'padrao_principal': padrao_principal,
            'padroes_secundarios': padroes_secundarios,
            'parametros': {
                'horarios_esperados': horarios,
                'tolerancia_minutos': tolerancia_minutos,
                'dias_consecutivos_minimo': dias_consecutivos,
                'concentracao_horaria': concentracao,
                'confianca_padrao': padrao_principal['confianca']
            },
            'criterio_reincidencia': {
                'descricao': f"Considerar REINCIDENTE se ocorrer {dias_consecutivos}+ dias consecutivos",
                'condicao': f"Nos horários: {', '.join([f'{h:02d}:00' for h in horarios])} (±{tolerancia_minutos}min)",
                'janela': f"Últimos {dias_consecutivos} dias"
            },
            'implementacao': {
                'logica': f"""
                FOR cada novo alerta:
                    hora_alerta = extrair_hora(alerta)
                    
                    # Verificar se está em horário de pico
                    em_horario_pico = FALSE
                    FOR horario_esperado in {horarios}:
                        IF abs(hora_alerta - horario_esperado) <= {tolerancia_minutos / 60:.2f}:
                            em_horario_pico = TRUE
                            BREAK
                    
                    IF em_horario_pico:
                        ultimos_dias = buscar_ultimos_dias({dias_consecutivos})
                        
                        dias_com_padrao = 0
                        FOR dia in ultimos_dias:
                            IF dia_tem_alerta_em_horario_pico(dia):
                                dias_com_padrao += 1
                        
                        IF dias_com_padrao >= {dias_consecutivos}:
                            MARCAR_COMO_REINCIDENTE()
                """
            }
        }
    
    elif tipo == 'DIAS_FIXOS':
        dias = padrao_principal['dias']
        concentracao = padrao_principal['concentracao']
        
        dias_map = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 
                    4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
        dias_nome = [dias_map[d] for d in dias]
        
        semanas_consecutivas = 3 if concentracao > 75 else 4
        
        return {
            'tipo': 'DIAS_RECORRENTES',
            'subtipo': 'SEMANAL',
            'padrao_principal': padrao_principal,
            'padroes_secundarios': padroes_secundarios,
            'parametros': {
                'dias_esperados': dias,
                'dias_nome': dias_nome,
                'semanas_consecutivas_minimo': semanas_consecutivas,
                'concentracao_semanal': concentracao,
                'confianca_padrao': padrao_principal['confianca']
            },
            'criterio_reincidencia': {
                'descricao': f"Considerar REINCIDENTE se ocorrer {semanas_consecutivas}+ semanas consecutivas",
                'condicao': f"Nos dias: {', '.join(dias_nome)}",
                'janela': f"Últimas {semanas_consecutivas} semanas"
            },
            'implementacao': {
                'logica': f"""
                FOR cada novo alerta:
                    dia_semana = extrair_dia_semana(alerta)
                    
                    IF dia_semana in {dias}:
                        ultimas_semanas = buscar_ultimas_semanas({semanas_consecutivas})
                        
                        semanas_com_padrao = 0
                        FOR semana in ultimas_semanas:
                            IF semana_tem_alerta_em_dias_pico(semana):
                                semanas_com_padrao += 1
                        
                        IF semanas_com_padrao >= {semanas_consecutivas}:
                            MARCAR_COMO_REINCIDENTE()
                """
            }
        }
    
    elif tipo == 'BURST':
        n_bursts = padrao_principal['num_bursts']
        
        # Para bursts, usar intervalo curto
        media_intervalo = results['basic_stats']['mean']
        intervalo_burst = media_intervalo / 3  # Um terço do intervalo médio
        
        ocorrencias_minimas_burst = 5
        janela_horas = max(6, intervalo_burst * ocorrencias_minimas_burst * 1.5)
        
        return {
            'tipo': 'RAJADA',
            'subtipo': 'BURST',
            'padrao_principal': padrao_principal,
            'padroes_secundarios': padroes_secundarios,
            'parametros': {
                'intervalo_burst_horas': round(intervalo_burst, 2),
                'ocorrencias_minimas_burst': ocorrencias_minimas_burst,
                'janela_horas': round(janela_horas, 2),
                'num_bursts_historico': n_bursts,
                'confianca_padrao': padrao_principal['confianca']
            },
            'criterio_reincidencia': {
                'descricao': f"Considerar REINCIDENTE se {ocorrencias_minimas_burst}+ alertas em janela curta",
                'condicao': f"Intervalo entre alertas < {intervalo_burst:.1f}h",
                'janela': f"Janela de {janela_horas:.0f}h"
            },
            'implementacao': {
                'logica': f"""
                FOR cada novo alerta:
                    ultimos_alertas = buscar_ultimos_alertas(janela={janela_horas:.0f}h)
                    
                    IF len(ultimos_alertas) >= {ocorrencias_minimas_burst}:
                        intervalos = calcular_intervalos(ultimos_alertas)
                        
                        em_burst = TRUE
                        FOR intervalo in intervalos:
                            IF intervalo > {intervalo_burst:.1f}:
                                em_burst = FALSE
                                BREAK
                        
                        IF em_burst:
                            MARCAR_COMO_REINCIDENTE()
                            PRIORIDADE_MAXIMA()
                """
            }
        }
    
    elif tipo == 'REGULAR_SEM_CICLO':
        cv = padrao_principal['cv']
        
        # Padrão regular mas sem ciclo claro - usar estatísticas
        media = results['basic_stats']['mean']
        std = results['basic_stats']['std']
        
        intervalo_esperado_min = max(0.5, media - std)
        intervalo_esperado_max = media + std
        
        ocorrencias_minimas = 4
        janela = (intervalo_esperado_max * ocorrencias_minimas) * 1.5
        
        return {
            'tipo': 'ESTATISTICO',
            'subtipo': 'REGULAR_SEM_CICLO',
            'padrao_principal': padrao_principal,
            'padroes_secundarios': padroes_secundarios,
            'parametros': {
                'intervalo_medio_horas': round(media, 2),
                'desvio_padrao_horas': round(std, 2),
                'intervalo_min_horas': round(intervalo_esperado_min, 2),
                'intervalo_max_horas': round(intervalo_esperado_max, 2),
                'ocorrencias_consecutivas_minimas': ocorrencias_minimas,
                'janela_analise_horas': round(janela, 2),
                'cv': cv,
                'confianca_padrao': padrao_principal['confianca']
            },
            'criterio_reincidencia': {
                'descricao': f"Considerar REINCIDENTE se {ocorrencias_minimas}+ ocorrências regulares",
                'condicao': f"Com intervalo entre {intervalo_esperado_min:.1f}h e {intervalo_esperado_max:.1f}h",
                'janela': f"Últimas {janela:.0f}h"
            },
            'implementacao': {
                'logica': f"""
                FOR cada novo alerta:
                    ultimos_alertas = buscar_ultimos_alertas(janela={janela:.0f}h)
                    
                    IF len(ultimos_alertas) >= {ocorrencias_minimas}:
                        intervalos = calcular_intervalos(ultimos_alertas)
                        
                        dentro_do_padrao = 0
                        FOR intervalo in intervalos:
                            IF {intervalo_esperado_min:.1f} <= intervalo <= {intervalo_esperado_max:.1f}:
                                dentro_do_padrao += 1
                        
                        IF dentro_do_padrao >= {ocorrencias_minimas - 1}:
                            MARCAR_COMO_REINCIDENTE()
                """
            }
        }
    
    else:  # SEM_PADRAO_CLARO ou outros
        return gerar_regra_generica()


def gerar_regra_generica():
    """Regra genérica quando não há padrão claro"""
    return {
        'tipo': 'THRESHOLD_GENERICO',
        'subtipo': 'SEM_PADRAO',
        'padrao_principal': {
            'tipo': 'SEM_PADRAO_CLARO',
            'descricao': 'Sem padrão recorrente detectável',
            'confianca': 50
        },
        'padroes_secundarios': [],
        'parametros': {
            'ocorrencias_janela': 3,
            'janela_horas': 24,
            'metodo': 'Threshold simples'
        },
        'criterio_reincidencia': {
            'descricao': "Manter regra atual de threshold",
            'condicao': "3 ou mais alertas",
            'janela': "Janela de 24 horas"
        },
        'implementacao': {
            'logica': """
            FOR cada novo alerta:
                ultimos_24h = buscar_ultimos_alertas(janela=24h)
                
                IF len(ultimos_24h) >= 3:
                    MARCAR_COMO_REINCIDENTE()
            """
        }
    }


def calcular_efetividade_regra(df, regra, regra_atual={'ocorrencias': 3, 'janela_horas': 24}):
    """
    Simula a efetividade da regra customizada vs regra atual
    
    Returns:
        dict: Métricas de comparação
    """
    if len(df) < 3:
        return {
            'regra_customizada': {'deteccoes': 0, 'taxa': 0},
            'regra_atual': {'deteccoes': 0, 'taxa': 0},
            'melhoria': 0
        }
    
    df_sorted = df.sort_values('created_on').reset_index(drop=True)
    
    # Simular regra ATUAL (3 em 24h)
    deteccoes_atual = 0
    for i in range(len(df_sorted)):
        janela_inicio = df_sorted.loc[i, 'created_on'] - timedelta(hours=regra_atual['janela_horas'])
        alertas_na_janela = df_sorted[
            (df_sorted['created_on'] >= janela_inicio) & 
            (df_sorted['created_on'] <= df_sorted.loc[i, 'created_on'])
        ]
        if len(alertas_na_janela) >= regra_atual['ocorrencias']:
            deteccoes_atual += 1
    
    # Simular regra CUSTOMIZADA
    deteccoes_custom = 0
    tipo_regra = regra['tipo']
    params = regra['parametros']
    
    if tipo_regra == 'INTERVALO_FIXO':
        periodo = params['periodo_esperado_horas']
        tolerancia = params['tolerancia_horas']
        ocorrencias_min = params['ocorrencias_consecutivas_minimas']
        
        for i in range(ocorrencias_min - 1, len(df_sorted)):
            ultimos = df_sorted.iloc[max(0, i - ocorrencias_min + 1):i + 1]
            intervalos = ultimos['created_on'].diff().dt.total_seconds() / 3600
            intervalos = intervalos.dropna()
            
            if len(intervalos) >= ocorrencias_min - 1:
                no_padrao = sum((periodo - tolerancia <= iv <= periodo + tolerancia) for iv in intervalos)
                if no_padrao >= ocorrencias_min - 1:
                    deteccoes_custom += 1
    
    elif tipo_regra == 'HORARIOS_RECORRENTES':
        horarios = params['horarios_esperados']
        tolerancia_h = params['tolerancia_minutos'] / 60
        dias_consecutivos = params['dias_consecutivos_minimo']
        
        for i in range(len(df_sorted)):
            hora_atual = df_sorted.loc[i, 'created_on'].hour + df_sorted.loc[i, 'created_on'].minute / 60
            
            # Verificar se está em horário de pico
            em_horario_pico = any(abs(hora_atual - h) <= tolerancia_h for h in horarios)
            
            if em_horario_pico:
                data_atual = df_sorted.loc[i, 'created_on'].date()
                datas_anteriores = [data_atual - timedelta(days=d) for d in range(1, dias_consecutivos)]
                
                dias_com_padrao = 1  # Dia atual
                for data in datas_anteriores:
                    alertas_dia = df_sorted[df_sorted['created_on'].dt.date == data]
                    if len(alertas_dia) > 0:
                        for _, alerta in alertas_dia.iterrows():
                            h = alerta['created_on'].hour + alerta['created_on'].minute / 60
                            if any(abs(h - hp) <= tolerancia_h for hp in horarios):
                                dias_com_padrao += 1
                                break
                
                if dias_com_padrao >= dias_consecutivos:
                    deteccoes_custom += 1
    
    elif tipo_regra == 'DIAS_RECORRENTES':
        dias_esperados = params['dias_esperados']
        semanas_min = params['semanas_consecutivas_minimo']
        
        for i in range(len(df_sorted)):
            dia_semana_atual = df_sorted.loc[i, 'created_on'].dayofweek
            
            if dia_semana_atual in dias_esperados:
                data_atual = df_sorted.loc[i, 'created_on']
                
                semanas_com_padrao = 1
                for semana_offset in range(1, semanas_min):
                    data_semana_anterior = data_atual - timedelta(weeks=semana_offset)
                    inicio_semana = data_semana_anterior - timedelta(days=3)
                    fim_semana = data_semana_anterior + timedelta(days=3)
                    
                    alertas_semana = df_sorted[
                        (df_sorted['created_on'] >= inicio_semana) & 
                        (df_sorted['created_on'] <= fim_semana)
                    ]
                    
                    if any(a.dayofweek in dias_esperados for a in alertas_semana['created_on']):
                        semanas_com_padrao += 1
                
                if semanas_com_padrao >= semanas_min:
                    deteccoes_custom += 1
    
    elif tipo_regra == 'RAJADA':
        janela_h = params['janela_horas']
        occ_min = params['ocorrencias_minimas_burst']
        
        for i in range(len(df_sorted)):
            janela_inicio = df_sorted.loc[i, 'created_on'] - timedelta(hours=janela_h)
            alertas_na_janela = df_sorted[
                (df_sorted['created_on'] >= janela_inicio) & 
                (df_sorted['created_on'] <= df_sorted.loc[i, 'created_on'])
            ]
            if len(alertas_na_janela) >= occ_min:
                deteccoes_custom += 1
    
    else:  # THRESHOLD_GENERICO ou outros
        deteccoes_custom = deteccoes_atual
    
    # Calcular taxas
    taxa_atual = (deteccoes_atual / len(df_sorted)) * 100 if len(df_sorted) > 0 else 0
    taxa_custom = (deteccoes_custom / len(df_sorted)) * 100 if len(df_sorted) > 0 else 0
    
    melhoria = taxa_custom - taxa_atual
    
    return {
        'regra_customizada': {
            'deteccoes': deteccoes_custom,
            'taxa': round(taxa_custom, 1)
        },
        'regra_atual': {
            'deteccoes': deteccoes_atual,
            'taxa': round(taxa_atual, 1)
        },
        'melhoria': round(melhoria, 1),
        'total_alertas': len(df_sorted)
    }


# ============================================================
# FUNÇÕES AUXILIARES PARA MULTIPROCESSING
# ============================================================

def analyze_single_short_ci_recurrence(short_ci, df_original):
    """Função auxiliar para análise de reincidência de um único short_ci (para multiprocessing)"""
    try:
        df_ci = df_original[df_original['short_ci'] == short_ci].copy()
        df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
        df_ci = df_ci.dropna(subset=['created_on'])
        df_ci = df_ci.sort_values('created_on')
        
        if len(df_ci) < 3:
            return {
                'short_ci': short_ci,
                'total_occurrences': len(df_ci),
                'score': 0,
                'classification': '⚪ DADOS INSUFICIENTES',
                'mean_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'predictability_score': 0
            }
        
        analyzer = AdvancedRecurrenceAnalyzer(df_ci, short_ci)
        return analyzer.analyze_silent()
    
    except Exception as e:
        return {
            'short_ci': short_ci,
            'total_occurrences': 0,
            'score': 0,
            'classification': f'⚪ ERRO: {str(e)[:50]}',
            'mean_interval_hours': None,
            'cv': None,
            'regularity_score': 0,
            'periodicity_detected': False,
            'predictability_score': 0
        }


def analyze_chunk_recurrence(short_ci_list, df_original):
    """Processa um chunk de short_ci para análise de reincidência"""
    results = []
    for short_ci in short_ci_list:
        result = analyze_single_short_ci_recurrence(short_ci, df_original)
        if result:
            results.append(result)
    return results


# ============================================================
# CLASSE DE ANÁLISE DE REINCIDÊNCIA
# ============================================================

class AdvancedRecurrenceAnalyzer:
    """Analisador completo de padrões de reincidência"""
    
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id
        self.cache = {}
    
    def _prepare_data(self):
        """Preparação vetorizada dos dados"""
        if self.df is None or len(self.df) < 3:
            return None
        
        df = self.df.sort_values('created_on').copy()
        
        # Vetorizar operações
        df['timestamp'] = df['created_on'].astype('int64') // 10**9
        df['time_diff_seconds'] = df['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        df['time_diff_days'] = df['time_diff_seconds'] / 86400
        
        # Componentes temporais
        dt = df['created_on'].dt
        if 'hour' not in df.columns:
            df['hour'] = dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = dt.dayofweek
        if 'day_of_month' not in df.columns:
            df['day_of_month'] = dt.day
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = dt.isocalendar().week
        if 'month' not in df.columns:
            df['month'] = dt.month
        if 'day_name' not in df.columns:
            df['day_name'] = dt.day_name()
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        if 'is_business_hours' not in df.columns:
            df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] <= 17)
        
        return df
    
    def analyze(self):
        """Método principal com interface Streamlit COMPLETA"""
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
        
        # Executar TODAS as análises
        results = {}
        results['basic_stats'] = self._analyze_basic_statistics(intervals_hours)
        results['regularity'] = self._analyze_regularity(intervals_hours)
        results['periodicity'] = self._analyze_periodicity(intervals_hours)
        results['autocorr'] = self._analyze_autocorrelation(intervals_hours)
        results['temporal'] = self._analyze_temporal_patterns(df)
        results['clusters'] = self._analyze_clusters(df, intervals_hours)
        results['bursts'] = self._detect_bursts(intervals_hours)
        results['seasonality'] = self._analyze_seasonality(df)
        results['changepoints'] = self._detect_changepoints(intervals_hours)
        results['anomalies'] = self._detect_anomalies(intervals_hours)
        results['predictability'] = self._calculate_predictability(intervals_hours)
        results['stability'] = self._analyze_stability(intervals_hours, df)
        results['contextual'] = self._analyze_contextual_dependencies(df)
        results['vulnerability'] = self._identify_vulnerability_windows(df, intervals_hours)
        results['maturity'] = self._analyze_pattern_maturity(df, intervals_hours)
        results['prediction_confidence'] = self._calculate_prediction_confidence(intervals_hours)
        results['markov'] = self._analyze_markov_chains(intervals_hours)
        results['randomness'] = self._advanced_randomness_tests(intervals_hours)
        
        # Classificação final COM REGRAS CUSTOMIZADAS
        self._final_classification(results, df, intervals_hours)
    
    def analyze_silent(self):
        """Análise silenciosa para processamento em lote"""
        df = self._prepare_data()
        if df is None or len(df) < 3:
            return None
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            return None
        
        # Executar análises essenciais sem interface
        results = {}
        results['basic_stats'] = self._analyze_basic_statistics_silent(intervals_hours)
        results['regularity'] = self._analyze_regularity_silent(intervals_hours)
        results['periodicity'] = self._analyze_periodicity_silent(intervals_hours)
        results['autocorr'] = self._analyze_autocorrelation_silent(intervals_hours)
        results['predictability'] = self._calculate_predictability_silent(intervals_hours)
        results['markov'] = self._analyze_markov_chains_silent(intervals_hours)
        results['randomness'] = self._advanced_randomness_tests_silent(intervals_hours)
        results['stability'] = self._analyze_stability_silent(intervals_hours)
        results['temporal'] = self._analyze_temporal_patterns_silent(df)
        results['bursts'] = self._detect_bursts_silent(intervals_hours)
        results['contextual'] = self._analyze_contextual_dependencies_silent(df)
        
        # Calcular score final
        final_score, classification = self._calculate_final_score(results)
        
        return {
            'short_ci': self.alert_id,
            'total_occurrences': len(df),
            'score': final_score,
            'classification': classification,
            'mean_interval_hours': results['basic_stats']['mean'],
            'median_interval_hours': results['basic_stats']['median'],
            'cv': results['basic_stats']['cv'],
            'regularity_score': results['regularity']['regularity_score'],
            'periodicity_detected': results['periodicity']['has_strong_periodicity'],
            'dominant_period_hours': results['periodicity'].get('dominant_period_hours'),
            'predictability_score': results['predictability']['predictability_score'],
            'next_occurrence_prediction_hours': results['predictability']['next_expected_hours']
        }
    
    # ============================================================
    # ANÁLISES COM INTERFACE (versão completa)
    # ============================================================
    
    def _analyze_basic_statistics(self, intervals):
        """Estatísticas básicas"""
        st.subheader("📊 1. Estatísticas de Intervalos")
        
        stats_dict = {
            'mean': np.mean(intervals),
            'median': np.median(intervals),
            'std': np.std(intervals),
            'min': np.min(intervals),
            'max': np.max(intervals),
            'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'),
            'q25': np.percentile(intervals, 25),
            'q75': np.percentile(intervals, 75),
            'iqr': np.percentile(intervals, 75) - np.percentile(intervals, 25)
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
        col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
        col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
        col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
        col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
        
        return stats_dict
    
    def _analyze_regularity(self, intervals):
        """Análise de regularidade"""
        st.subheader("🎯 2. Regularidade")
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        if cv < 0.20:
            regularity_score = 95
            pattern_type = "🟢 ALTAMENTE REGULAR"
            pattern_color = "green"
        elif cv < 0.40:
            regularity_score = 80
            pattern_type = "🟢 REGULAR"
            pattern_color = "lightgreen"
        elif cv < 0.70:
            regularity_score = 60
            pattern_type = "🟡 SEMI-REGULAR"
            pattern_color = "yellow"
        elif cv < 1.20:
            regularity_score = 35
            pattern_type = "🟠 IRREGULAR"
            pattern_color = "orange"
        else:
            regularity_score = 15
            pattern_type = "🔴 MUITO IRREGULAR"
            pattern_color = "red"
        
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
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': pattern_color},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgray"},
                        {'range': [35, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key='reg_gauge')
        
        return {'cv': cv, 'regularity_score': regularity_score, 'type': pattern_type}
    
    def _analyze_periodicity(self, intervals):
        """Análise de periodicidade com FFT"""
        st.subheader("🔍 3. Periodicidade (FFT)")
        
        if len(intervals) < 10:
            st.info("📊 Mínimo de 10 intervalos necessários")
            return {'has_periodicity': False, 'has_strong_periodicity': False}
        
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
            dominant_periods = 1 / dominant_freqs
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            
            if len(dominant_periods) > 0:
                has_strong_periodicity = True
                dominant_period_hours = dominant_periods[0] * np.mean(intervals)
                
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
        fig.update_layout(
            title="Espectro de Frequência",
            xaxis_title="Período",
            yaxis_title="Magnitude",
            height=300,
            xaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True, key='fft')
        
        return {
            'periods': dominant_periods,
            'has_periodicity': len(dominant_periods) > 0,
            'has_strong_periodicity': has_strong_periodicity,
            'dominant_period_hours': dominant_period_hours
        }
    
    def _analyze_autocorrelation(self, intervals):
        """Análise de autocorrelação"""
        st.subheader("📈 4. Autocorrelação")
        
        if len(intervals) < 5:
            return {}
        
        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        lags = np.arange(len(autocorr))
        threshold = 2 / np.sqrt(len(intervals))
        
        significant_peaks = [(i, autocorr[i]) for i in range(1, min(len(autocorr), 20)) 
                           if autocorr[i] > threshold]
        
        if significant_peaks:
            st.success("✅ **Autocorrelação Significativa:**")
            for lag, corr in significant_peaks[:3]:
                st.write(f"• Lag {lag}: {corr:.2f}")
        else:
            st.info("📊 Sem autocorrelação significativa")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lags[:min(30, len(lags))],
            y=autocorr[:min(30, len(autocorr))],
            mode='lines+markers'
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red")
        fig.update_layout(title="Autocorrelação", height=300)
        st.plotly_chart(fig, use_container_width=True, key='autocorr')
        
        return {'peaks': significant_peaks, 'has_autocorr': len(significant_peaks) > 0}
    
    def _analyze_temporal_patterns(self, df):
        """Análise de padrões temporais"""
        st.subheader("⏰ 5. Padrões Temporais")
        
        hourly = df.groupby('hour').size()
        hourly = hourly.reindex(range(24), fill_value=0)
        
        daily = df.groupby('day_of_week').size()
        daily = daily.reindex(range(7), fill_value=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Bar(
                x=list(range(24)),
                y=hourly.values,
                marker_color=['red' if v > hourly.mean() + hourly.std() else 'lightblue' 
                            for v in hourly.values]
            ))
            fig.update_layout(title="Por Hora", xaxis_title="Hora", height=250)
            st.plotly_chart(fig, use_container_width=True, key='hourly')
            
            peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist()
            if peak_hours:
                st.success(f"🕐 **Picos:** {', '.join([f'{h:02d}:00' for h in peak_hours])}")
        
        with col2:
            days_map = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
            fig = go.Figure(go.Bar(
                x=days_map,
                y=daily.values,
                marker_color=['red' if v > daily.mean() + daily.std() else 'lightgreen' 
                            for v in daily.values]
            ))
            fig.update_layout(title="Por Dia", xaxis_title="Dia", height=250)
            st.plotly_chart(fig, use_container_width=True, key='daily')
            
            peak_days = daily[daily > daily.mean() + daily.std()].index.tolist()
            if peak_days:
                st.success(f"📅 **Picos:** {', '.join([days_map[d] for d in peak_days])}")
        
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
        
        hourly_conc = hourly_pct.nlargest(3).sum() if len(hourly_pct) > 0 else 0
        daily_conc = daily_pct.nlargest(3).sum() if len(daily_pct) > 0 else 0
        
        return {
            'hourly_concentration': hourly_conc,
            'daily_concentration': daily_conc,
            'peak_hours': peak_hours,
            'peak_days': peak_days
        }
    
    def _analyze_clusters(self, df, intervals):
        """Detecção de clusters temporais"""
        st.subheader("🎯 6. Clusters Temporais")
        
        if len(df) < 10:
            st.info("Mínimo de 10 ocorrências necessário")
            return {}
        
        first_ts = df['timestamp'].min()
        time_features = ((df['timestamp'] - first_ts) / 3600).values.reshape(-1, 1)
        
        eps = np.median(intervals) * 2
        dbscan = DBSCAN(eps=eps, min_samples=3)
        clusters = dbscan.fit_predict(time_features)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Clusters", n_clusters)
        col2.metric("📊 Em Clusters", len(clusters) - n_noise)
        col3.metric("🔴 Isolados", n_noise)
        
        if n_clusters > 0:
            st.success(f"✅ **{n_clusters} clusters** identificados")
        
        return {'n_clusters': n_clusters, 'n_noise': n_noise}
    
    def _detect_bursts(self, intervals):
        """Detecção de bursts"""
        st.subheader("💥 7. Detecção de Bursts")
        
        burst_threshold = np.percentile(intervals, 25)
        
        is_burst = intervals < burst_threshold
        burst_changes = np.diff(np.concatenate(([False], is_burst, [False])))
        burst_starts = np.where(burst_changes == 1)[0]
        burst_ends = np.where(burst_changes == -1)[0]
        
        burst_sequences = [(start, end) for start, end in zip(burst_starts, burst_ends) 
                          if end - start >= 3]
        
        col1, col2 = st.columns(2)
        col1.metric("💥 Bursts", len(burst_sequences))
        
        if burst_sequences:
            avg_size = np.mean([end - start for start, end in burst_sequences])
            col2.metric("📊 Tamanho Médio", f"{avg_size:.1f}")
            st.warning(f"⚠️ **{len(burst_sequences)} bursts** detectados")
        else:
            st.success("✅ Sem padrão de rajadas")
        
        return {'n_bursts': len(burst_sequences), 'has_bursts': len(burst_sequences) > 0}
    
    def _analyze_seasonality(self, df):
        """Análise de sazonalidade"""
        st.subheader("🌡️ 8. Sazonalidade")
        
        date_range = (df['created_on'].max() - df['created_on'].min()).days
        
        if date_range < 30:
            st.info("📊 Período curto para análise sazonal")
            return {}
        
        weekly = df.groupby('week_of_year').size()
        
        if len(weekly) >= 4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly.index,
                y=weekly.values,
                mode='lines+markers',
                fill='tozeroy'
            ))
            fig.update_layout(title="Evolução Semanal", height=250)
            st.plotly_chart(fig, use_container_width=True, key='weekly')
            
            if len(weekly) > 3:
                slope, _, _, p_value, _ = stats.linregress(weekly.index.values, weekly.values)
                if p_value < 0.05:
                    if slope > 0:
                        st.warning("📈 **Tendência crescente**")
                        return {'trend': 'increasing', 'slope': slope}
                    else:
                        st.success("📉 **Tendência decrescente**")
                        return {'trend': 'decreasing', 'slope': slope}
        
        return {'trend': 'stable'}
    
    def _detect_changepoints(self, intervals):
        """Detecção de pontos de mudança"""
        st.subheader("🔀 9. Pontos de Mudança")
        
        if len(intervals) < 20:
            st.info("Mínimo de 20 intervalos necessário")
            return {}
        
        cumsum = np.cumsum(intervals - np.mean(intervals))
        
        window = 5
        changes = []
        for i in range(window, len(cumsum) - window):
            before = np.mean(intervals[max(0, i-window):i])
            after = np.mean(intervals[i:min(len(intervals), i+window)])
            if abs(before - after) > np.std(intervals):
                changes.append(i)
        
        filtered = []
        for cp in changes:
            if not filtered or cp - filtered[-1] > 5:
                filtered.append(cp)
        
        if filtered:
            st.warning(f"⚠️ **{len(filtered)} pontos de mudança** detectados")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(cumsum))), y=cumsum, mode='lines'))
            for cp in filtered:
                fig.add_vline(x=cp, line_dash="dash", line_color="red")
            fig.update_layout(title="CUSUM", height=250)
            st.plotly_chart(fig, use_container_width=True, key='cusum')
        else:
            st.success("✅ Comportamento estável")
        
        return {'changepoints': filtered, 'has_changes': len(filtered) > 0}
    
    def _detect_anomalies(self, intervals):
        """Detecção de anomalias"""
        st.subheader("🚨 10. Detecção de Anomalias")
        
        z_scores = np.abs(stats.zscore(intervals))
        z_anomalies = np.sum(z_scores > 3)
        
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_anomalies = np.sum((intervals < lower) | (intervals > upper))
        
        iso_anomalies = 0
        if len(intervals) >= 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(intervals.reshape(-1, 1))
            iso_anomalies = np.sum(predictions == -1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Z-Score", f"{z_anomalies}")
        col2.metric("IQR", f"{iqr_anomalies}")
        col3.metric("Iso. Forest", f"{iso_anomalies}")
        
        total_anomalies = max(z_anomalies, iqr_anomalies, iso_anomalies)
        anomaly_rate = total_anomalies / len(intervals) * 100
        
        if anomaly_rate > 10:
            st.warning(f"⚠️ **{anomaly_rate:.1f}%** de anomalias")
        else:
            st.success("✅ Baixa taxa de anomalias")
        
        return {'anomaly_rate': anomaly_rate, 'total_anomalies': total_anomalies}
    
    def _calculate_predictability(self, intervals):
        """Score de previsibilidade"""
        st.subheader("🔮 11. Previsibilidade")
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
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
        
        mean_interval = np.mean(intervals)
        
        col1, col2 = st.columns(2)
        col1.metric("Score", f"{predictability}%")
        col2.metric("Próxima Ocorrência", f"{mean_interval:.1f}h")
        
        if predictability > 70:
            st.success("✅ Altamente previsível")
        elif predictability > 50:
            st.info("📊 Moderadamente previsível")
        else:
            st.warning("⚠️ Pouco previsível")
        
        return {'predictability_score': predictability, 'next_expected_hours': mean_interval}
    
    def _analyze_stability(self, intervals, df):
        """Análise de estabilidade"""
        st.subheader("📊 12. Estabilidade")
        
        if len(intervals) < 10:
            return {'is_stable': True, 'stability_score': 50}
        
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]
        
        _, p_value = stats.ttest_ind(first_half, second_half)
        is_stable = p_value > 0.05
        
        mean_diff = abs(np.mean(second_half) - np.mean(first_half))
        drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
        
        stability_score = max(0, 100 - drift_pct)
        
        col1, col2 = st.columns(2)
        col1.metric("Score", f"{stability_score:.1f}%")
        col2.metric("Drift", f"{drift_pct:.1f}%")
        
        if is_stable and drift_pct < 20:
            st.success("✅ Padrão estável")
        elif drift_pct < 50:
            st.info("📊 Moderadamente estável")
        else:
            st.warning("⚠️ Padrão instável")
        
        return {'is_stable': is_stable, 'stability_score': stability_score, 'drift_pct': drift_pct}
    
    def _analyze_contextual_dependencies(self, df):
        """Análise de dependências contextuais"""
        st.subheader("🌐 13. Dependências Contextuais")
        
        try:
            br_holidays = holidays.Brazil(years=df['created_on'].dt.year.unique())
            df['is_holiday'] = df['created_on'].dt.date.apply(lambda x: x in br_holidays)
        except:
            df['is_holiday'] = False
        
        business_days = df[~df['is_weekend'] & ~df['is_holiday']]
        weekend_days = df[df['is_weekend']]
        holiday_days = df[df['is_holiday']]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Dias Úteis", f"{len(business_days)/len(df)*100:.1f}%")
        col2.metric("🎉 Fins de Semana", f"{len(weekend_days)/len(df)*100:.1f}%")
        col3.metric("🎊 Feriados", f"{len(holiday_days)/len(df)*100:.1f}%")
        
        if len(holiday_days) > 0:
            st.warning(f"⚠️ {len(holiday_days)} alertas em feriados")
        
        return {
            'holiday_correlation': len(holiday_days) / len(df) if len(df) > 0 else 0,
            'weekend_correlation': len(weekend_days) / len(df) if len(df) > 0 else 0
        }
    
    def _identify_vulnerability_windows(self, df, intervals):
        """Janelas de vulnerabilidade"""
        st.subheader("🎯 14. Janelas de Vulnerabilidade")
        
        vulnerability_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        vulnerability_matrix['risk_score'] = (
            vulnerability_matrix['count'] / vulnerability_matrix['count'].max() * 100
        )
        
        top_windows = vulnerability_matrix.nlargest(5, 'risk_score')
        
        day_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        
        st.write("**🔴 Top 5 Janelas Críticas:**")
        for idx, row in top_windows.iterrows():
            day = day_map[row['day_of_week']]
            hour = int(row['hour'])
            risk = row['risk_score']
            st.write(f"• **{day} {hour:02d}:00** - Score: {risk:.1f} ({row['count']} alertas)")
        
        return {'top_windows': top_windows.to_dict('records')}
    
    def _analyze_pattern_maturity(self, df, intervals):
        """Maturidade do padrão"""
        st.subheader("📈 15. Maturidade do Padrão")
        
        n_periods = 4
        period_size = len(intervals) // n_periods
        
        if period_size < 2:
            st.info("Período insuficiente")
            return {}
        
        periods_stats = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(intervals)
            period_intervals = intervals[start:end]
            
            periods_stats.append({
                'period': i + 1,
                'mean': np.mean(period_intervals),
                'cv': np.std(period_intervals) / np.mean(period_intervals) if np.mean(period_intervals) > 0 else 0
            })
        
        periods_df = pd.DataFrame(periods_stats)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods_df['period'],
            y=periods_df['cv'],
            mode='lines+markers',
            name='CV',
            line=dict(color='red', width=3)
        ))
        fig.update_layout(
            title="Evolução da Variabilidade",
            xaxis_title="Período",
            yaxis_title="CV",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key='maturity')
        
        slope = np.polyfit(periods_df['period'], periods_df['cv'], 1)[0]
        
        if slope < -0.05:
            st.success("✅ **Amadurecendo**: Variabilidade decrescente")
            maturity = "maturing"
        elif slope > 0.05:
            st.warning("⚠️ **Degradando**: Variabilidade crescente")
            maturity = "degrading"
        else:
            st.info("📊 **Estável**: Variabilidade constante")
            maturity = "stable"
        
        return {'maturity': maturity, 'slope': slope}
    
    def _calculate_prediction_confidence(self, intervals):
        """Confiança da predição"""
        st.subheader("🎯 16. Confiança de Predição")
        
        if len(intervals) < 10:
            return {'confidence': 'low', 'score': 0}
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        n_samples = len(intervals)
        
        regularity_score = max(0, 100 - cv * 100)
        sample_score = min(100, (n_samples / 50) * 100)
        
        mid = len(intervals) // 2
        var1 = np.var(intervals[:mid])
        var2 = np.var(intervals[mid:])
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0
        stationarity_score = var_ratio * 100
        
        confidence_score = (regularity_score * 0.5 + sample_score * 0.3 + stationarity_score * 0.2)
        
        if confidence_score > 70:
            confidence = 'high'
        elif confidence_score > 40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        col1, col2 = st.columns(2)
        col1.metric("Confiança", confidence.upper())
        col2.metric("Score", f"{confidence_score:.1f}%")
        
        return {'confidence': confidence, 'score': confidence_score}
    
    def _analyze_markov_chains(self, intervals):
        """Cadeias de Markov"""
        st.subheader("🔗 17. Cadeias de Markov")
        
        if len(intervals) < 20:
            st.info("Mínimo de 20 intervalos necessário")
            return {}
        
        q25, q50, q75 = np.percentile(intervals, [25, 50, 75])
        
        def interval_to_state(val):
            if val <= q25:
                return 'Muito Curto'
            elif val <= q50:
                return 'Curto'
            elif val <= q75:
                return 'Normal'
            else:
                return 'Longo'
        
        states = [interval_to_state(i) for i in intervals]
        state_labels = ['Muito Curto', 'Curto', 'Normal', 'Longo']
        
        n_states = len(state_labels)
        transition_matrix = np.zeros((n_states, n_states))
        state_to_idx = {state: idx for idx, state in enumerate(state_labels)}
        
        for i in range(len(states) - 1):
            from_state = state_to_idx[states[i]]
            to_state = state_to_idx[states[i + 1]]
            transition_matrix[from_state, to_state] += 1
        
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transition_matrix / row_sums
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_probs,
            x=state_labels,
            y=state_labels,
            text=np.round(transition_probs, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Matriz de Transição",
            xaxis_title="Estado Seguinte",
            yaxis_title="Estado Atual",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key='markov_matrix')
        
        max_probs = transition_probs.max(axis=1)
        markov_score = np.mean(max_probs) * 100
        
        st.metric("Score Markoviano", f"{markov_score:.1f}%")
        
        if markov_score > 60:
            st.success("✅ Forte padrão markoviano")
        elif markov_score > 30:
            st.info("📊 Padrão moderado")
        else:
            st.warning("⚠️ Padrão fraco")
        
        return {'markov_score': markov_score}
    
    def _advanced_randomness_tests(self, intervals):
        """Testes de aleatoriedade"""
        st.subheader("🎲 18. Testes de Aleatoriedade")
        
        if len(intervals) < 10:
            st.info("Mínimo de 10 intervalos necessário")
            return {}
        
        results = {}
        
        # Runs Test
        st.write("**1️⃣ Runs Test**")
        median = np.median(intervals)
        runs = np.diff(intervals > median).sum() + 1
        expected_runs = len(intervals) / 2
        
        col1, col2 = st.columns(2)
        col1.metric("Runs Observados", runs)
        col2.metric("Runs Esperados", f"{expected_runs:.1f}")
        
        # Permutation Entropy
        st.write("**2️⃣ Permutation Entropy**")
        
        def permutation_entropy(series, order=3):
            n = len(series)
            permutations = []
            
            for i in range(n - order + 1):
                pattern = series[i:i+order]
                sorted_idx = np.argsort(pattern)
                perm = tuple(sorted_idx)
                permutations.append(perm)
            
            perm_counts = Counter(permutations)
            probs = np.array(list(perm_counts.values())) / len(permutations)
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(math.factorial(order))
            return entropy / max_entropy if max_entropy > 0 else 0
        
        if len(intervals) >= 10:
            perm_entropy = permutation_entropy(intervals)
            complexity = perm_entropy * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Entropia", f"{perm_entropy:.3f}")
            col2.metric("Complexidade", f"{complexity:.1f}%")
            
            if complexity > 70:
                st.success("✅ Alta complexidade")
            else:
                st.warning("⚠️ Baixa complexidade")
        
        # Hurst Exponent
        st.write("**3️⃣ Hurst Exponent**")
        
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
        
        if len(intervals) >= 20:
            hurst = hurst_exponent(intervals)
            
            if hurst is not None:
                st.metric("Hurst", f"{hurst:.3f}")
                
                if hurst < 0.45:
                    st.info("📉 Anti-persistente")
                elif hurst > 0.55:
                    st.warning("📈 Persistente")
                else:
                    st.success("🎲 Random Walk")
                
                results['hurst'] = hurst
        
        # Score final
        st.markdown("---")
        randomness_score = 50  # Simplificado
        st.metric("Score de Aleatoriedade", f"{randomness_score:.0f}%")
        
        if randomness_score >= 60:
            st.success("✅ Comportamento aleatório")
        elif randomness_score >= 40:
            st.info("📊 Comportamento misto")
        else:
            st.warning("⚠️ Comportamento determinístico")
        
        results['overall_randomness_score'] = randomness_score
        return results
    
    # ============================================================
    # MÉTODOS SILENCIOSOS (para batch processing)
    # ============================================================
    
    def _analyze_basic_statistics_silent(self, intervals):
        return {
            'mean': np.mean(intervals),
            'median': np.median(intervals),
            'std': np.std(intervals),
            'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        }
    
    def _analyze_regularity_silent(self, intervals):
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        if cv < 0.20:
            regularity_score = 95
        elif cv < 0.40:
            regularity_score = 80
        elif cv < 0.70:
            regularity_score = 60
        elif cv < 1.20:
            regularity_score = 35
        else:
            regularity_score = 15
        return {'cv': cv, 'regularity_score': regularity_score}
    
    def _analyze_periodicity_silent(self, intervals):
        if len(intervals) < 4:
            return {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}
        try:
            N = len(intervals)
            yf = fft(intervals)
            xf = fftfreq(N, d=1)[:N//2]
            power = 2.0/N * np.abs(yf[:N//2])
            if len(power) > 1:
                peak_idx = np.argmax(power[1:]) + 1
                dominant_freq = xf[peak_idx]
                dominant_period = 1/dominant_freq if dominant_freq != 0 else None
                mean_interval = np.mean(intervals)
                dominant_period_hours = dominant_period * mean_interval if dominant_period else None
                peak_power = power[peak_idx]
                mean_power = np.mean(power)
                strength_ratio = peak_power / mean_power if mean_power > 0 else 0
                has_strong = strength_ratio > 3.0
                has_moderate = 1.5 < strength_ratio <= 3.0
                return {
                    'has_strong_periodicity': has_strong,
                    'has_moderate_periodicity': has_moderate,
                    'dominant_period_hours': dominant_period_hours
                }
        except:
            pass
        return {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}
    
    def _analyze_autocorrelation_silent(self, intervals):
        if len(intervals) < 5:
            return {'max_autocorr': 0}
        try:
            max_lag = min(len(intervals) // 2, 20)
            autocorr_values = []
            for lag in range(1, max_lag + 1):
                if lag < len(intervals):
                    corr = np.corrcoef(intervals[:-lag], intervals[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            return {'max_autocorr': max(autocorr_values) if autocorr_values else 0}
        except:
            return {'max_autocorr': 0}
    
    def _calculate_predictability_silent(self, intervals):
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
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
        mean_interval = np.mean(intervals)
        return {'predictability_score': predictability, 'next_expected_hours': mean_interval}
    
    def _analyze_markov_chains_silent(self, intervals):
        if len(intervals) < 5:
            return {'markov_score': 0}
        try:
            bins = np.percentile(intervals, [0, 33, 67, 100])
            states = np.digitize(intervals, bins[1:-1])
            n_states = 3
            transition_matrix = np.zeros((n_states, n_states))
            for i in range(len(states) - 1):
                current = states[i]
                next_state = states[i + 1]
                transition_matrix[current, next_state] += 1
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_matrix / row_sums
            max_probs = transition_matrix.max(axis=1)
            markov_score = np.mean(max_probs) * 100
            return {'markov_score': markov_score}
        except:
            return {'markov_score': 0}
    
    def _advanced_randomness_tests_silent(self, intervals):
        if len(intervals) < 5:
            return {'overall_randomness_score': 50}
        try:
            median = np.median(intervals)
            runs = np.diff(intervals > median).sum() + 1
            expected_runs = len(intervals) / 2
            runs_score = min(abs(runs - expected_runs) / expected_runs * 100, 100)
            overall_randomness = runs_score
            return {'overall_randomness_score': overall_randomness}
        except:
            return {'overall_randomness_score': 50}
    
    def _analyze_stability_silent(self, intervals):
        if len(intervals) < 10:
            return {'is_stable': True, 'stability_score': 50}
        try:
            mid = len(intervals) // 2
            first_half = intervals[:mid]
            second_half = intervals[mid:]
            _, p_value = stats.ttest_ind(first_half, second_half)
            is_stable = p_value > 0.05
            mean_diff = abs(np.mean(second_half) - np.mean(first_half))
            drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
            stability_score = max(0, 100 - drift_pct)
            return {'is_stable': is_stable, 'stability_score': stability_score}
        except:
            return {'is_stable': True, 'stability_score': 50}
    
    def _analyze_temporal_patterns_silent(self, df):
        hourly = df.groupby('hour').size()
        daily = df.groupby('day_of_week').size()
        
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
        
        hourly_conc = hourly_pct.nlargest(3).sum() if len(hourly_pct) > 0 else 0
        daily_conc = daily_pct.nlargest(3).sum() if len(daily_pct) > 0 else 0
        
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist() if len(hourly) > 0 else []
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist() if len(daily) > 0 else []
        
        return {
            'hourly_concentration': hourly_conc,
            'daily_concentration': daily_conc,
            'peak_hours': peak_hours,
            'peak_days': peak_days
        }
    
    def _detect_bursts_silent(self, intervals):
        if len(intervals) < 5:
            return {'n_bursts': 0, 'has_bursts': False}
        
        burst_threshold = np.percentile(intervals, 25)
        is_burst = intervals < burst_threshold
        burst_changes = np.diff(np.concatenate(([False], is_burst, [False])))
        burst_starts = np.where(burst_changes == 1)[0]
        burst_ends = np.where(burst_changes == -1)[0]
        
        burst_sequences = [(start, end) for start, end in zip(burst_starts, burst_ends) 
                          if end - start >= 3]
        
        return {'n_bursts': len(burst_sequences), 'has_bursts': len(burst_sequences) > 0}
    
    def _analyze_contextual_dependencies_silent(self, df):
        try:
            br_holidays = holidays.Brazil(years=df['created_on'].dt.year.unique())
            df['is_holiday'] = df['created_on'].dt.date.apply(lambda x: x in br_holidays)
        except:
            df['is_holiday'] = False
        
        weekend_days = df[df['is_weekend']]
        holiday_days = df[df['is_holiday']]
        
        return {
            'holiday_correlation': len(holiday_days) / len(df) if len(df) > 0 else 0,
            'weekend_correlation': len(weekend_days) / len(df) if len(df) > 0 else 0
        }
    
    # ============================================================
    # CLASSIFICAÇÃO FINAL COM REGRAS CUSTOMIZADAS
    # ============================================================
    
    def _calculate_final_score(self, results):
        """Calcula score final - CRITÉRIOS REVISADOS"""
        scores = {
            'regularity': results['regularity']['regularity_score'] * 0.25,
            'periodicity': (
                100 if results['periodicity']['has_strong_periodicity'] else
                50 if results['periodicity'].get('has_moderate_periodicity', False) else
                0
            ) * 0.25,
            'predictability': results['predictability']['predictability_score'] * 0.20,
            'determinism': (100 - results['randomness']['overall_randomness_score']) * 0.15,
            'autocorrelation': (results['autocorr']['max_autocorr'] * 100) * 0.10,
            'stability': results.get('stability', {}).get('stability_score', 50) * 0.05
        }
        
        final_score = sum(scores.values())
        
        # Thresholds REVISADOS (mais rigorosos)
        if final_score >= 75:
            classification = "🔴 REINCIDENTE CRÍTICO (P1)"
        elif final_score >= 55:
            classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
        elif final_score >= 35:
            classification = "🟡 PADRÃO DETECTÁVEL (P3)"
        else:
            classification = "🟢 NÃO REINCIDENTE (P4)"
        
        return round(final_score, 2), classification
    
    def _final_classification(self, results, df, intervals):
        """Classificação final COM REGRAS CUSTOMIZADAS"""
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL")
        
        final_score, classification = self._calculate_final_score(results)
        
        if final_score >= 75:
            level = "CRÍTICO"
            color = "red"
            priority = "P1"
            recommendation = "**Ação Imediata:** Criar automação, runbook e investigar causa raiz"
        elif final_score >= 55:
            level = "ALTO"
            color = "orange"
            priority = "P2"
            recommendation = "**Ação Recomendada:** Monitorar evolução e considerar automação"
        elif final_score >= 35:
            level = "MÉDIO"
            color = "yellow"
            priority = "P3"
            recommendation = "**Ação Sugerida:** Documentar padrão e revisar thresholds"
        else:
            level = "BAIXO"
            color = "green"
            priority = "P4"
            recommendation = "**Ação:** Análise caso a caso"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
            st.metric("Score de Reincidência", f"{final_score:.0f}/100")
            
            st.markdown("#### 📊 Breakdown")
            breakdown = {
                'Regularidade (25%)': results['regularity']['regularity_score'] * 0.25,
                'Periodicidade (25%)': (100 if results['periodicity'].get('has_strong_periodicity', False) else 0) * 0.25,
                'Previsibilidade (20%)': results['predictability']['predictability_score'] * 0.20,
                'Determinismo (15%)': (100 - results['randomness']['overall_randomness_score']) * 0.15,
                'Autocorrelação (10%)': (results['autocorr'].get('has_autocorr', False) * 100) * 0.10,
                'Estabilidade (5%)': results['stability'].get('stability_score', 50) * 0.05
            }
            
            for criterion, points in breakdown.items():
                st.write(f"• {criterion}: **{points:.1f} pts**")
            
            st.info(recommendation)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score,
                title={'text': "Score Final"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgray"},
                        {'range': [35, 55], 'color': "lightyellow"},
                        {'range': [55, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key='final_gauge')
        
        # ============================================================
        # NOVA SEÇÃO: GERAÇÃO DE REGRAS CUSTOMIZADAS
        # ============================================================
        
        st.markdown("---")
        st.header("📋 REGRA CUSTOMIZADA SUGERIDA")
        
        # Gerar regra customizada
        regra = gerar_regra_customizada(results, df, intervals)
        
        # Mostrar padrão principal
        padrao_principal = regra['padrao_principal']
        st.success(f"**🎯 Padrão Detectado:** {padrao_principal['tipo']}")
        st.write(f"**Descrição:** {padrao_principal['descricao']}")
        st.write(f"**Confiança:** {padrao_principal['confianca']}%")
        
        # Padrões secundários
        padroes_secundarios = regra.get('padroes_secundarios', [])
        if padroes_secundarios:
            st.info("**📊 Padrões Secundários Detectados:**")
            for padrao in padroes_secundarios:
                st.write(f"• {padrao['tipo']}: {padrao['descricao']} (confiança: {padrao['confianca']}%)")
        
        # Parâmetros da regra
        st.markdown("---")
        st.subheader("⚙️ Parâmetros da Regra")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Critério de Reincidência:**")
            criterio = regra['criterio_reincidencia']
            st.write(f"• {criterio['descricao']}")
            st.write(f"• {criterio['condicao']}")
            st.write(f"• Janela: {criterio['janela']}")
        
        with col2:
            st.markdown("**🔧 Parâmetros Técnicos:**")
            params = regra['parametros']
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    st.write(f"• {key}: **{value:.2f}**" if isinstance(value, float) else f"• {key}: **{value}**")
                else:
                    st.write(f"• {key}: **{value}**")
        
        # Implementação
        st.markdown("---")
        st.subheader("💻 Lógica de Implementação")
        
        with st.expander("Ver pseudocódigo", expanded=False):
            st.code(regra['implementacao']['logica'], language='python')
        
        # Comparação com regra atual
        st.markdown("---")
        st.subheader("📊 Comparação: Regra Customizada vs. Regra Atual")
        
        efetividade = calcular_efetividade_regra(df, regra)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Regra Atual (3 em 24h)",
                f"{efetividade['regra_atual']['taxa']}%",
                f"{efetividade['regra_atual']['deteccoes']} detecções"
            )
        
        with col2:
            st.metric(
                "Regra Customizada",
                f"{efetividade['regra_customizada']['taxa']}%",
                f"{efetividade['regra_customizada']['deteccoes']} detecções"
            )
        
        with col3:
            delta = efetividade['melhoria']
            delta_color = "normal" if delta >= 0 else "inverse"
            st.metric(
                "Melhoria",
                f"{abs(delta):.1f}%",
                f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}%",
                delta_color=delta_color
            )
        
        if efetividade['melhoria'] > 10:
            st.success(f"✅ A regra customizada é **{efetividade['melhoria']:.1f}% mais efetiva** que a regra atual!")
        elif efetividade['melhoria'] > 0:
            st.info(f"📊 A regra customizada apresenta melhoria modesta de **{efetividade['melhoria']:.1f}%**")
        elif efetividade['melhoria'] < -10:
            st.warning(f"⚠️ A regra atual parece mais adequada (diferença: {abs(efetividade['melhoria']):.1f}%)")
        else:
            st.info("📊 Ambas as regras têm efetividade similar")
        
        # Predição se score alto
        if final_score >= 55:
            st.markdown("---")
            st.subheader("🔮 Predição")
            
            last_alert = df['created_on'].max()
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            pred_time = last_alert + pd.Timedelta(hours=mean_interval)
            conf_interval = 1.96 * std_interval
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Próxima Ocorrência", pred_time.strftime('%d/%m %H:%M'))
            col2.metric("Intervalo", f"{mean_interval:.1f}h")
            col3.metric("Confiança (95%)", f"± {conf_interval:.1f}h")
        
        # Exportar
        st.markdown("---")
        export_data = {
            'short_ci': self.alert_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': final_score,
            'classificacao': classification,
            'nivel': level,
            'prioridade': priority,
            'cv': results['basic_stats']['cv'],
            'regularidade': results['regularity']['regularity_score'],
            'periodicidade': results['periodicity'].get('has_strong_periodicity', False),
            'previsibilidade': results['predictability']['predictability_score'],
            'padrao_tipo': padrao_principal['tipo'],
            'padrao_descricao': padrao_principal['descricao'],
            'padrao_confianca': padrao_principal['confianca'],
            'regra_tipo': regra['tipo'],
            'regra_descricao': regra['criterio_reincidencia']['descricao']
        }
        
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            "⬇️ Exportar Relatório Completo",
            csv,
            f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# FUNÇÕES AUXILIARES DE AGRUPAMENTO
# ============================================================

def identify_alert_groups(alert_data, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Identifica grupos de alertas contínuos"""
    if len(alert_data) == 0:
        return alert_data, []
    
    alert_data = alert_data.sort_values('created_on').reset_index(drop=True)
    alert_data['time_diff_hours'] = alert_data['created_on'].diff().dt.total_seconds() / 3600
    alert_data['date'] = alert_data['created_on'].dt.date
    unique_dates = alert_data['date'].nunique()
    
    if unique_dates == 1:
        alert_data['group_id'] = -1
        alert_data['is_isolated'] = True
        return alert_data, []
    
    alert_data['group_id'] = -1
    current_group = 0
    group_start_idx = 0
    
    for i in range(len(alert_data)):
        if i == 0:
            continue
        
        gap = alert_data.loc[i, 'time_diff_hours']
        
        if gap > max_gap_hours:
            group_size = i - group_start_idx
            if group_size >= min_group_size:
                alert_data.loc[group_start_idx:i-1, 'group_id'] = current_group
                current_group += 1
            group_start_idx = i
    
    group_size = len(alert_data) - group_start_idx
    if group_size >= min_group_size:
        alert_data.loc[group_start_idx:, 'group_id'] = current_group
    
    daily_counts = alert_data.groupby('date').size()
    avg_daily = daily_counts.mean()
    spike_threshold = avg_daily * spike_threshold_multiplier
    spike_dates = daily_counts[daily_counts > spike_threshold].index
    
    if len(spike_dates) > 0:
        alert_data.loc[alert_data['date'].isin(spike_dates), 'group_id'] = -1
    
    alert_data['is_isolated'] = alert_data['group_id'] == -1
    
    groups_info = []
    for group_id in alert_data[alert_data['group_id'] >= 0]['group_id'].unique():
        group_data = alert_data[alert_data['group_id'] == group_id]
        groups_info.append({
            'group_id': int(group_id),
            'size': len(group_data),
            'start_time': group_data['created_on'].min(),
            'end_time': group_data['created_on'].max(),
            'duration_hours': (group_data['created_on'].max() - group_data['created_on'].min()).total_seconds() / 3600
        })
    
    return alert_data, groups_info


def classify_alert_pattern(alert_data, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Classifica padrão do alerta"""
    n = len(alert_data)
    if n == 0:
        return {
            'pattern': 'isolated',
            'reason': 'Sem ocorrências',
            'occurrences': 0,
            'num_groups': 0,
            'isolated_occurrences': 0,
            'grouped_occurrences': 0,
            'groups_info': [],
            'unique_days': 0
        }
    
    unique_days = alert_data['created_on'].dt.date.nunique()
    
    if unique_days == 1:
        return {
            'pattern': 'isolated',
            'reason': f'Todos em 1 dia',
            'occurrences': n,
            'num_groups': 0,
            'isolated_occurrences': n,
            'grouped_occurrences': 0,
            'groups_info': [],
            'unique_days': 1
        }
    
    alert_data_processed, groups_info = identify_alert_groups(
        alert_data, max_gap_hours, min_group_size, spike_threshold_multiplier
    )
    
    num_groups = len(groups_info)
    isolated_count = alert_data_processed['is_isolated'].sum()
    grouped_count = n - isolated_count
    isolated_pct = (isolated_count / n) * 100
    
    if num_groups == 0:
        pattern = 'isolated'
        reason = f'Nenhum grupo'
    elif isolated_pct > 70:
        pattern = 'isolated'
        reason = f'{isolated_pct:.0f}% isolados'
    elif num_groups >= 2:
        pattern = 'continuous'
        reason = f'{num_groups} grupos'
    else:
        pattern = 'isolated'
        reason = f'Inconsistente'
    
    return {
        'pattern': pattern,
        'reason': reason,
        'occurrences': n,
        'num_groups': num_groups,
        'isolated_occurrences': int(isolated_count),
        'grouped_occurrences': int(grouped_count),
        'groups_info': groups_info,
        'unique_days': unique_days
    }


def process_single_alert(alert_id, df_original, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Processa um único alerta para análise global"""
    try:
        df_alert = df_original[df_original['short_ci'] == alert_id].copy()
        if len(df_alert) < 1:
            return None
        
        pattern_info = classify_alert_pattern(df_alert, max_gap_hours, min_group_size, spike_threshold_multiplier)
        
        df_alert['hour'] = df_alert['created_on'].dt.hour
        df_alert['day_of_week'] = df_alert['created_on'].dt.dayofweek
        df_alert['is_weekend'] = df_alert['day_of_week'].isin([5, 6])
        df_alert['is_business_hours'] = (df_alert['hour'] >= 9) & (df_alert['hour'] <= 17)
        df_alert = df_alert.sort_values('created_on')
        intervals_hours = df_alert['created_on'].diff().dt.total_seconds() / 3600
        intervals_hours = intervals_hours.dropna()
        
        period_days = (df_alert['created_on'].max() - df_alert['created_on'].min()).days + 1
        
        metrics = {
            'alert_id': alert_id,
            'pattern_type': pattern_info['pattern'],
            'pattern_reason': pattern_info['reason'],
            'total_ocorrencias': pattern_info['occurrences'],
            'num_grupos': pattern_info['num_groups'],
            'alertas_isolados': pattern_info['isolated_occurrences'],
            'alertas_agrupados': pattern_info['grouped_occurrences'],
            'pct_isolados': (pattern_info['isolated_occurrences'] / pattern_info['occurrences'] * 100) 
                           if pattern_info['occurrences'] > 0 else 0,
            'unique_days': pattern_info['unique_days'],
            'periodo_dias': period_days,
            'freq_dia': len(df_alert) / period_days if period_days > 0 else 0,
            'freq_semana': (len(df_alert) / period_days * 7) if period_days > 0 else 0,
            'freq_mes': (len(df_alert) / period_days * 30) if period_days > 0 else 0,
            'intervalo_medio_h': intervals_hours.mean() if len(intervals_hours) > 0 else None,
            'intervalo_mediano_h': intervals_hours.median() if len(intervals_hours) > 0 else None,
            'intervalo_std_h': intervals_hours.std() if len(intervals_hours) > 0 else None,
            'intervalo_min_h': intervals_hours.min() if len(intervals_hours) > 0 else None,
            'intervalo_max_h': intervals_hours.max() if len(intervals_hours) > 0 else None,
            'hora_pico': df_alert['hour'].mode().iloc[0] if len(df_alert['hour'].mode()) > 0 else 12,
            'pct_fins_semana': df_alert['is_weekend'].mean() * 100,
            'pct_horario_comercial': df_alert['is_business_hours'].mean() * 100,
            'variabilidade_intervalo': intervals_hours.std() / intervals_hours.mean() 
                                      if len(intervals_hours) > 0 and intervals_hours.mean() > 0 else 0,
            'primeiro_alerta': df_alert['created_on'].min(),
            'ultimo_alerta': df_alert['created_on'].max()
        }
        return metrics
    except Exception:
        return None


def process_alert_chunk(alert_ids, df_original, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Processa chunk de alertas"""
    return [metrics for alert_id in alert_ids 
            if (metrics := process_single_alert(alert_id, df_original, max_gap_hours, min_group_size, spike_threshold_multiplier))]


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class StreamlitAlertAnalyzer:
    def __init__(self):
        self.df_original = None
        self.df_all_alerts = None
        self.df = None
        self.dates = None
        self.alert_id = None
        self.max_gap_hours = 24
        self.min_group_size = 3
        self.spike_threshold_multiplier = 5
        self.groups_info = []

    def load_data(self, uploaded_file):
        """Carrega dados do CSV"""
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado: {len(df_raw)} registros")
            
            with st.expander("📋 Preview"):
                st.write(f"**Colunas:** {list(df_raw.columns)}")
                st.dataframe(df_raw.head())
            
            if 'created_on' not in df_raw.columns or 'short_ci' not in df_raw.columns:
                st.error("❌ Colunas obrigatórias: 'created_on' e 'short_ci'")
                return False
            
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on'])
            df_raw = df_raw.sort_values(['short_ci', 'created_on']).reset_index(drop=True)
            
            self.df_original = df_raw
            st.sidebar.write(f"**IDs:** {len(df_raw['short_ci'].unique())}")
            return True
        
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            return False

    def prepare_individual_analysis(self, alert_id):
        """Prepara análise individual"""
        df_filtered = self.df_original[self.df_original['short_ci'] == alert_id].copy()
        if len(df_filtered) == 0:
            return False

        df_filtered['date'] = df_filtered['created_on'].dt.date
        df_filtered['hour'] = df_filtered['created_on'].dt.hour
        df_filtered['day_of_week'] = df_filtered['created_on'].dt.dayofweek
        df_filtered['day_name'] = df_filtered['created_on'].dt.day_name()
        df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])
        df_filtered['is_business_hours'] = (df_filtered['hour'] >= 9) & (df_filtered['hour'] <= 17)
        df_filtered['time_diff_hours'] = df_filtered['created_on'].diff().dt.total_seconds() / 3600

        df_filtered, groups_info = identify_alert_groups(
            df_filtered, self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
        )

        self.df = df_filtered
        self.dates = df_filtered['created_on']
        self.alert_id = alert_id
        self.groups_info = groups_info
        return True

    def prepare_global_analysis(self, use_multiprocessing=True, max_gap_hours=24, 
                               min_group_size=3, spike_threshold_multiplier=5):
        """Prepara análise global COM MULTIPROCESSING"""
        st.header("🌍 Análise Global")
        
        self.df = self.df_original.copy()
        self.max_gap_hours = max_gap_hours
        self.min_group_size = min_group_size
        self.spike_threshold_multiplier = spike_threshold_multiplier
        
        unique_ids = self.df_original['short_ci'].unique()
        total_ids = len(unique_ids)
        st.info(f"📊 Processando {total_ids} Alert IDs...")
        
        alert_metrics = []
        
        if use_multiprocessing and total_ids > 10:
            n_processes = min(cpu_count(), total_ids, 8)
            st.write(f"🚀 Usando {n_processes} processos")
            
            chunk_size = max(1, total_ids // n_processes)
            id_chunks = [unique_ids[i:i + chunk_size] for i in range(0, total_ids, chunk_size)]
            
            progress_bar = st.progress(0)
            
            process_func = partial(
                process_alert_chunk,
                df_original=self.df_original,
                max_gap_hours=max_gap_hours,
                min_group_size=min_group_size,
                spike_threshold_multiplier=spike_threshold_multiplier
            )
            
            try:
                with Pool(processes=n_processes) as pool:
                    results = pool.map(process_func, id_chunks)
                    for result in results:
                        alert_metrics.extend(result)
                progress_bar.progress(1.0)
                st.success(f"✅ {len(alert_metrics)} processados")
            except Exception as e:
                st.error(f"❌ Erro: {e}")
                use_multiprocessing = False
                alert_metrics = []
        
        if not use_multiprocessing or len(alert_metrics) == 0:
            progress_bar = st.progress(0)
            for i, alert_id in enumerate(unique_ids):
                progress_bar.progress((i + 1) / total_ids)
                metrics = process_single_alert(
                    alert_id, self.df_original, 
                    max_gap_hours, min_group_size, spike_threshold_multiplier
                )
                if metrics:
                    alert_metrics.append(metrics)
            progress_bar.empty()
        
        self.df_all_alerts = pd.DataFrame(alert_metrics)
        return len(self.df_all_alerts) > 0

    def batch_analyze_all_short_ci_with_multiprocessing(self, progress_bar=None):
        """Análise de reincidência em lote COM MULTIPROCESSING"""
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None
            
            short_ci_list = self.df_original['short_ci'].unique()
            total = len(short_ci_list)
            
            use_mp = total > 20
            
            if use_mp:
                n_processes = min(cpu_count(), total, 8)
                st.info(f"🚀 Usando {n_processes} processos para {total} alertas")
                
                chunk_size = max(1, total // n_processes)
                chunks = [short_ci_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
                
                process_func = partial(analyze_chunk_recurrence, df_original=self.df_original)
                
                try:
                    all_results = []
                    with Pool(processes=n_processes) as pool:
                        for idx, chunk_results in enumerate(pool.imap(process_func, chunks)):
                            all_results.extend(chunk_results)
                            if progress_bar:
                                progress = (len(all_results) / total)
                                progress_bar.progress(progress, text=f"{len(all_results)}/{total}")
                    
                    return pd.DataFrame(all_results)
                
                except Exception as e:
                    st.warning(f"⚠️ Erro: {e}. Modo sequencial...")
                    use_mp = False
            
            if not use_mp:
                all_results = []
                for idx, short_ci in enumerate(short_ci_list):
                    if progress_bar:
                        progress_bar.progress((idx + 1) / total, text=f"{idx + 1}/{total}")
                    
                    result = analyze_single_short_ci_recurrence(short_ci, self.df_original)
                    if result:
                        all_results.append(result)
                
                return pd.DataFrame(all_results)
        
        except Exception as e:
            st.error(f"Erro: {e}")
            return None

    def complete_analysis_all_short_ci(self, progress_bar=None):
        """Análise COMPLETA COM MULTIPROCESSING"""
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None
            
            # 1. Análise global
            if progress_bar:
                progress_bar.progress(0.05, text="Análise global...")
            
            alert_ids = self.df_original['short_ci'].unique()
            
            use_mp_global = len(alert_ids) > 20
            
            if use_mp_global:
                n_processes = min(cpu_count(), len(alert_ids), 8)
                chunk_size = max(1, len(alert_ids) // n_processes)
                chunks = [alert_ids[i:i + chunk_size] for i in range(0, len(alert_ids), chunk_size)]
                
                process_func = partial(
                    process_alert_chunk,
                    df_original=self.df_original,
                    max_gap_hours=self.max_gap_hours,
                    min_group_size=self.min_group_size,
                    spike_threshold_multiplier=self.spike_threshold_multiplier
                )
                
                results_global = []
                try:
                    with Pool(processes=n_processes) as pool:
                        for chunk_result in pool.map(process_func, chunks):
                            results_global.extend(chunk_result)
                except:
                    for alert_id in alert_ids:
                        metrics = process_single_alert(
                            alert_id, self.df_original,
                            self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
                        )
                        if metrics:
                            results_global.append(metrics)
            else:
                results_global = []
                for alert_id in alert_ids:
                    metrics = process_single_alert(
                        alert_id, self.df_original,
                        self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
                    )
                    if metrics:
                        results_global.append(metrics)
            
            df_global = pd.DataFrame(results_global)
            df_global = df_global.rename(columns={'alert_id': 'short_ci'})
            
            if progress_bar:
                progress_bar.progress(0.20, text="Global concluída!")
            
            # 2. Análise de reincidência
            if progress_bar:
                progress_bar.progress(0.25, text="Análise de reincidência...")
            
            df_reincidencia = self.batch_analyze_all_short_ci_with_multiprocessing(progress_bar)
            
            if df_reincidencia is None or len(df_reincidencia) == 0:
                st.error("❌ Erro na reincidência")
                return None
            
            # 3. Merge
            if progress_bar:
                progress_bar.progress(0.95, text="Consolidando...")
            
            df_consolidated = pd.merge(
                df_global,
                df_reincidencia,
                on='short_ci',
                how='outer'
            )
            
            # Reordenar
            priority_columns = [
                'short_ci',
                'score',
                'classification',
                'pattern_type',
                'total_ocorrencias',
                'num_grupos',
                'alertas_isolados',
                'alertas_agrupados'
            ]
            
            other_columns = [col for col in df_consolidated.columns if col not in priority_columns]
            final_columns = [col for col in priority_columns + other_columns if col in df_consolidated.columns]
            
            df_consolidated = df_consolidated[final_columns]
            df_consolidated = df_consolidated.sort_values('score', ascending=False)
            
            if progress_bar:
                progress_bar.progress(1.0, text="✅ Completa!")
            
            return df_consolidated
        
        except Exception as e:
            st.error(f"Erro: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def show_basic_stats(self):
        """Estatísticas básicas"""
        st.header("📊 Estatísticas Básicas")
        
        total = len(self.df)
        period_days = (self.dates.max() - self.dates.min()).days + 1
        avg_per_day = total / period_days
        unique_days = self.df['date'].nunique()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔥 Total", total)
        col2.metric("📅 Período", period_days)
        col3.metric("📆 Dias Únicos", unique_days)
        col4.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        col5.metric("🕐 Último", self.dates.max().strftime("%d/%m %H:%M"))
        
        if unique_days == 1:
            st.warning("⚠️ Todos em 1 dia - ISOLADO")
        
        # Frequências
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
        
        # Intervalos
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            st.markdown("---")
            st.subheader("⏱️ Intervalos")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Média (h)", f"{intervals.mean():.2f}")
            col2.metric("Mediana (h)", f"{intervals.median():.2f}")
            col3.metric("Mínimo (h)", f"{intervals.min():.2f}")
            col4.metric("Máximo (h)", f"{intervals.max():.2f}")

    def show_individual_alert_analysis(self):
        """Análise individual do alerta"""
        st.header(f"📌 Análise: {self.alert_id}")
        
        if self.df is None or len(self.df) == 0:
            st.info("Sem dados")
            return
        
        unique_days = self.df['date'].nunique()
        is_single_day = unique_days == 1
        
        df_isolated = self.df[self.df['is_isolated']]
        df_grouped = self.df[~self.df['is_isolated']]
        
        st.subheader("📊 Estatísticas")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total", len(self.df))
        col2.metric("🔴 Isolados", len(df_isolated))
        col3.metric("🟢 Agrupados", len(df_grouped))
        col4.metric("📦 Grupos", len(self.groups_info))
        col5.metric("% Isolados", f"{(len(df_isolated)/len(self.df)*100):.1f}%")
        col6.metric("Dias", unique_days)
        
        if is_single_day:
            st.warning("⚠️ Todos em 1 dia - ISOLADO")


# ============================================================
# FUNÇÃO MAIN
# ============================================================

def main():
    st.title("🚨 Analisador de Alertas - Completo + Regras Customizadas")
    st.markdown("### Com multiprocessing, 18 análises e geração automática de regras")
    
    st.sidebar.header("⚙️ Configurações")
    
    with st.sidebar.expander("🎛️ Parâmetros", expanded=False):
        max_gap_hours = st.slider("Gap Máximo (h)", 1, 72, 24)
        min_group_size = st.slider("Tamanho Mínimo", 2, 10, 3)
        spike_threshold = st.slider("Multiplicador Spike", 2.0, 10.0, 5.0, 0.5)
    
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo",
        ["🔍 Individual", "🔄 Reincidência Global", "📊 Completa + CSV"]
    )
    
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])
    
    if uploaded_file:
        analyzer = StreamlitAlertAnalyzer()
        
        if analyzer.load_data(uploaded_file):
            
            if analysis_mode == "🔍 Individual":
                id_counts = analyzer.df_original['short_ci'].value_counts()
                id_options = [f"{uid} ({count})" for uid, count in id_counts.items()]
                selected = st.sidebar.selectbox("Short CI", id_options)
                selected_id = selected.split(" (")[0]
                
                if st.sidebar.button("🚀 Analisar", type="primary"):
                    analyzer.max_gap_hours = max_gap_hours
                    analyzer.min_group_size = min_group_size
                    analyzer.spike_threshold_multiplier = spike_threshold
                    
                    if analyzer.prepare_individual_analysis(selected_id):
                        st.success(f"Analisando: {selected_id}")
                        
                        tab1, tab2 = st.tabs(["📊 Básico", "🔄 Reincidência + Regras Customizadas"])
                        
                        with tab1:
                            analyzer.show_basic_stats()
                            analyzer.show_individual_alert_analysis()
                        
                        with tab2:
                            recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
                            recurrence_analyzer.analyze()
            
            elif analysis_mode == "🔄 Reincidência Global":
                st.subheader("🔄 Reincidência Global (COM MULTIPROCESSING)")
                
                if st.sidebar.button("🚀 Executar", type="primary"):
                    if analyzer.prepare_global_analysis():
                        num_ci = len(analyzer.df['short_ci'].unique())
                        st.info(f"📊 Analisando {num_ci} Short CIs...")
                        
                        progress_bar = st.progress(0)
                        results_df = analyzer.batch_analyze_all_short_ci_with_multiprocessing(progress_bar)
                        progress_bar.empty()
                        
                        if results_df is not None and len(results_df) > 0:
                            st.success(f"✅ {len(results_df)} analisados!")
                            
                            # Stats
                            st.subheader("📊 Resumo")
                            
                            critical = len(results_df[results_df['classification'].str.contains('CRÍTICO', na=False)])
                            high = len(results_df[results_df['classification'].str.contains('PARCIALMENTE', na=False)])
                            medium = len(results_df[results_df['classification'].str.contains('DETECTÁVEL', na=False)])
                            low = len(results_df[results_df['classification'].str.contains('NÃO', na=False)])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("🔴 P1", critical)
                            col2.metric("🟠 P2", high)
                            col3.metric("🟡 P3", medium)
                            col4.metric("🟢 P4", low)
                            
                            # Top 20
                            st.subheader("🏆 Top 20")
                            top_20 = results_df.nlargest(20, 'score')[[
                                'short_ci', 'total_occurrences', 'score', 'classification',
                                'cv', 'regularity_score', 'predictability_score'
                            ]].round(2)
                            st.dataframe(top_20, use_container_width=True)
                            
                            # Download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download",
                                csv,
                                f"reincidencia_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
            
            elif analysis_mode == "📊 Completa + CSV":
                st.subheader("📊 Completa COM MULTIPROCESSING")
                
                if st.sidebar.button("🚀 Executar", type="primary"):
                    if analyzer.prepare_global_analysis():
                        st.info("⏱️ Processando...")
                        
                        progress_bar = st.progress(0)
                        df_consolidated = analyzer.complete_analysis_all_short_ci(progress_bar)
                        progress_bar.empty()
                        
                        if df_consolidated is not None and len(df_consolidated) > 0:
                            st.success(f"✅ {len(df_consolidated)} processados!")
                            
                            # Resumo
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
                            
                            # Top 20
                            st.subheader("🏆 Top 20")
                            top_20 = df_consolidated.nlargest(20, 'score')[[
                                'short_ci', 'score', 'classification', 'pattern_type',
                                'total_ocorrencias', 'freq_dia'
                            ]].round(2)
                            st.dataframe(top_20, use_container_width=True)
                            
                            # Downloads
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
                            
                            summary = df_consolidated[['short_ci', 'score', 'classification']].copy()
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
            ### 🚀 Funcionalidades:
            
            **🔍 Análise Individual:**
            - 📊 Estatísticas básicas
            - 🔄 **18 análises avançadas de reincidência**
            - 📋 **GERAÇÃO AUTOMÁTICA DE REGRAS CUSTOMIZADAS**
            - ⚙️ **Parâmetros específicos por série**
            - 📊 **Comparação com regra atual (3 em 24h)**
            - 💻 **Pseudocódigo de implementação**
            
            **🔄 Reincidência Global:**
            - Análise em lote COM MULTIPROCESSING
            - Scores para todos os alertas
            - Download em CSV
            
            **📊 Completa:**
            - Global + Reincidência + Regras
            - CSV consolidado
            - Todas as métricas
            
            ### 📋 Colunas CSV:
            - `short_ci`: ID do alerta
            - `created_on`: Data/hora
            
            ### 🎯 Tipos de Regras Geradas:
            - **INTERVALO_FIXO:** Padrão periódico (8 em 8h, etc)
            - **HORARIOS_RECORRENTES:** Sempre nos mesmos horários
            - **DIAS_RECORRENTES:** Sempre nos mesmos dias da semana
            - **RAJADA:** Padrão de bursts
            - **ESTATISTICO:** Regular mas sem ciclo claro
            - **THRESHOLD_GENERICO:** Quando não há padrão
            
            ### ✨ Novo: Para cada série com padrão, você receberá:
            1. **Padrão detectado** com confiança
            2. **Regra customizada** com parâmetros específicos
            3. **Comparação** com regra atual (efetividade)
            4. **Pseudocódigo** para implementação
            """)


if __name__ == "__main__":
    main()
