# ============================================================================
# SIMULACIÓN DE DIAGNÓSTICO DE BATERÍA HÍBRIDA NiMH
# Tesis: Equipo modular para diagnóstico de SOH y retención de carga
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Crear carpeta para guardar resultados
if not os.path.exists('resultados'):
    os.makedirs('resultados')

print("="*60)
print("SIMULACIÓN DE DIAGNÓSTICO DE BATERÍA HÍBRIDA NiMH")
print("="*60)

# ============================================================================
# PARÁMETROS DE LA BATERÍA
# ============================================================================

# Configuración del pack
NUM_MODULOS = 14                    # Número de módulos en serie
VOLTAJE_NOMINAL_MODULO = 7.2        # Voltaje nominal por módulo (V)
CAPACIDAD_NOMINAL = 6.5             # Capacidad nominal (Ah)
R0_NOMINAL = 0.025                  # Resistencia interna nominal (Ohm) = 25 mΩ

# Parámetros del modelo RC de segundo orden
R1 = 0.010      # Resistencia de polarización 1 (Ohm)
R2 = 0.008      # Resistencia de polarización 2 (Ohm)
C1 = 1000       # Capacitancia 1 (F)
C2 = 5000       # Capacitancia 2 (F)
TAU1 = R1 * C1  # Constante de tiempo 1
TAU2 = R2 * C2  # Constante de tiempo 2

# Parámetros de simulación
CORRIENTE_DESCARGA = 3.25           # Corriente de descarga C/2 (A)
TIEMPO_TOTAL = 7200                 # Tiempo total de simulación (s) = 2 horas
DT = 1                              # Paso de tiempo (s)
VOLTAJE_MINIMO = 6.0                # Voltaje mínimo de corte por módulo (V)

# ============================================================================
# FUNCIÓN: Curva OCV (Open Circuit Voltage) vs SOC para NiMH
# ============================================================================

def calcular_ocv(soc):
    """
    Calcula el voltaje de circuito abierto en función del SOC.
    Curva típica para batería NiMH de 6 celdas (7.2V nominal).
    """
    # Polinomio ajustado para NiMH (valores típicos)
    ocv = 6.0 + 1.4 * soc + 0.3 * soc**2 - 0.5 * soc**3 + 0.2 * soc**4
    return ocv

# ============================================================================
# DEFINICIÓN DE ESCENARIOS DE DEGRADACIÓN
# ============================================================================

def crear_escenarios():
    """
    Define los 4 escenarios de prueba con diferentes estados de degradación.
    Retorna un diccionario con los parámetros de cada módulo por escenario.
    """
    escenarios = {}
    
    # ESCENARIO 1: Batería sana (todos los módulos en buen estado)
    escenarios['Escenario 1 - Batería Sana'] = {
        'descripcion': 'Todos los módulos en condiciones óptimas',
        'capacidades': [6.5] * NUM_MODULOS,  # Todos a 6.5 Ah
        'resistencias': [0.025] * NUM_MODULOS  # Todos a 25 mΩ
    }
    
    # ESCENARIO 2: Un módulo degradado (desbalance localizado)
    capacidades_esc2 = [6.5] * NUM_MODULOS
    resistencias_esc2 = [0.025] * NUM_MODULOS
    capacidades_esc2[6] = 4.0    # Módulo 7 degradado (índice 6)
    resistencias_esc2[6] = 0.050  # Resistencia aumentada a 50 mΩ
    
    escenarios['Escenario 2 - Un Módulo Degradado'] = {
        'descripcion': 'Módulo 7 con degradación severa (desbalance localizado)',
        'capacidades': capacidades_esc2,
        'resistencias': resistencias_esc2
    }
    
    # ESCENARIO 3: Varios módulos degradados (envejecimiento distribuido)
    capacidades_esc3 = [6.0, 6.0, 6.0, 6.0, 6.0,  # Módulos 1-5
                        4.5, 4.8, 4.5, 5.0,        # Módulos 6-9 (más degradados)
                        6.0, 6.0, 6.0, 6.0, 6.0]   # Módulos 10-14
    resistencias_esc3 = [0.030, 0.030, 0.030, 0.030, 0.030,
                         0.040, 0.045, 0.040, 0.038,
                         0.030, 0.030, 0.030, 0.030, 0.030]
    
    escenarios['Escenario 3 - Degradación Distribuida'] = {
        'descripcion': 'Envejecimiento general con módulos 6-9 más afectados',
        'capacidades': capacidades_esc3,
        'resistencias': resistencias_esc3
    }
    
    # ESCENARIO 4: Batería altamente degradada (fin de vida)
    escenarios['Escenario 4 - Batería Muy Degradada'] = {
        'descripcion': 'Pack cercano al fin de vida útil (SOH < 50%)',
        'capacidades': [3.0, 3.2, 2.8, 3.1, 3.0, 2.9, 3.0, 3.2, 2.8, 3.0, 3.1, 2.9, 3.0, 3.2],
        'resistencias': [0.055, 0.052, 0.058, 0.054, 0.056, 0.060, 0.055, 0.053, 0.059, 0.055, 0.054, 0.057, 0.055, 0.053]
    }
    
    return escenarios

# ============================================================================
# FUNCIÓN: Simulación de un módulo individual
# ============================================================================

def simular_modulo(capacidad, r0, tiempo_total, dt, corriente):
    """
    Simula el comportamiento de un módulo durante descarga.
    Retorna arrays de tiempo, voltaje, SOC.
    """
    # Inicialización
    n_puntos = int(tiempo_total / dt) + 1
    tiempo = np.linspace(0, tiempo_total, n_puntos)
    voltaje = np.zeros(n_puntos)
    soc = np.zeros(n_puntos)
    v1 = np.zeros(n_puntos)  # Voltaje en RC1
    v2 = np.zeros(n_puntos)  # Voltaje en RC2
    
    # Condiciones iniciales
    soc[0] = 1.0  # SOC inicial = 100%
    v1[0] = 0
    v2[0] = 0
    voltaje[0] = calcular_ocv(soc[0])
    
    # Simulación temporal
    for i in range(1, n_puntos):
        # Actualizar SOC (conteo de coulombs)
        delta_soc = (corriente * dt) / (capacidad * 3600)
        soc[i] = soc[i-1] - delta_soc
        
        # Limitar SOC entre 0 y 1
        if soc[i] < 0:
            soc[i] = 0
        
        # Actualizar voltajes de polarización (modelo RC)
        v1[i] = v1[i-1] * np.exp(-dt/TAU1) + R1 * corriente * (1 - np.exp(-dt/TAU1))
        v2[i] = v2[i-1] * np.exp(-dt/TAU2) + R2 * corriente * (1 - np.exp(-dt/TAU2))
        
        # Calcular voltaje del módulo
        ocv = calcular_ocv(soc[i])
        voltaje[i] = ocv - corriente * r0 - v1[i] - v2[i]
        
        # Verificar si llegó al voltaje mínimo
        if voltaje[i] < VOLTAJE_MINIMO:
            voltaje[i] = VOLTAJE_MINIMO
            # Marcar el resto como voltaje mínimo
            voltaje[i:] = VOLTAJE_MINIMO
            soc[i:] = soc[i]
            break
    
    return tiempo, voltaje, soc

# ============================================================================
# FUNCIÓN: Simulación completa del pack
# ============================================================================

def simular_pack(escenario_nombre, escenario_datos):
    """
    Simula todos los módulos del pack y calcula métricas de diagnóstico.
    """
    print(f"\n{'='*60}")
    print(f"Simulando: {escenario_nombre}")
    print(f"Descripción: {escenario_datos['descripcion']}")
    print(f"{'='*60}")
    
    capacidades = escenario_datos['capacidades']
    resistencias = escenario_datos['resistencias']
    
    # Arrays para almacenar resultados de todos los módulos
    resultados_modulos = []
    
    # Simular cada módulo
    for i in range(NUM_MODULOS):
        tiempo, voltaje, soc = simular_modulo(
            capacidad=capacidades[i],
            r0=resistencias[i],
            tiempo_total=TIEMPO_TOTAL,
            dt=DT,
            corriente=CORRIENTE_DESCARGA
        )
        
        # Calcular capacidad efectiva (Ah descargados hasta voltaje mínimo)
        # Encontrar el índice donde el voltaje llega al mínimo
        indices_minimo = np.where(voltaje <= VOLTAJE_MINIMO + 0.01)[0]
        if len(indices_minimo) > 0:
            tiempo_descarga = tiempo[indices_minimo[0]]
        else:
            tiempo_descarga = TIEMPO_TOTAL
        
        capacidad_efectiva = (CORRIENTE_DESCARGA * tiempo_descarga) / 3600
        
        # Calcular SOH
        soh = (capacidad_efectiva / CAPACIDAD_NOMINAL) * 100
        if soh > 100:
            soh = 100
        
        resultados_modulos.append({
            'modulo': i + 1,
            'tiempo': tiempo,
            'voltaje': voltaje,
            'soc': soc,
            'capacidad_nominal': CAPACIDAD_NOMINAL,
            'capacidad_real': capacidades[i],
            'capacidad_efectiva': capacidad_efectiva,
            'resistencia': resistencias[i] * 1000,  # Convertir a mΩ
            'soh': soh,
            'tiempo_descarga': tiempo_descarga
        })
        
        print(f"  Módulo {i+1:2d}: Cap.Efec={capacidad_efectiva:.2f} Ah, "
              f"R0={resistencias[i]*1000:.1f} mΩ, SOH={soh:.1f}%")
    
    return resultados_modulos

# ============================================================================
# FUNCIÓN: Análisis de desbalance
# ============================================================================

def analizar_desbalance(resultados_modulos, tiempo_analisis=1800):
    """
    Analiza el desbalance entre módulos en un instante dado.
    """
    # Encontrar índice del tiempo de análisis
    tiempo = resultados_modulos[0]['tiempo']
    idx = np.argmin(np.abs(tiempo - tiempo_analisis))
    
    # Obtener voltajes de todos los módulos en ese instante
    voltajes = [m['voltaje'][idx] for m in resultados_modulos]
    voltaje_promedio = np.mean(voltajes)
    voltaje_std = np.std(voltajes)
    
    # Clasificar módulos
    clasificacion = []
    for i, v in enumerate(voltajes):
        diferencia = abs(v - voltaje_promedio) * 1000  # en mV
        soh = resultados_modulos[i]['soh']
        
        if diferencia > 150 or soh < 60:
            estado = "DEGRADADO"
        elif diferencia > 100 or soh < 70:
            estado = "SOSPECHOSO"
        else:
            estado = "SANO"
        
        clasificacion.append({
            'modulo': i + 1,
            'voltaje': v,
            'diferencia_mV': diferencia,
            'soh': soh,
            'estado': estado
        })
    
    return clasificacion, voltaje_promedio, voltaje_std

# ============================================================================
# FUNCIÓN: Generar gráficos
# ============================================================================

def generar_graficos(escenario_nombre, resultados_modulos):
    """
    Genera y guarda los gráficos de resultados.
    """
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Resultados de Simulación: {escenario_nombre}', fontsize=14, fontweight='bold')
    
    # Colores para los módulos
    colores = plt.cm.tab20(np.linspace(0, 1, NUM_MODULOS))
    
    # -------- Gráfico 1: Voltaje vs Tiempo --------
    ax1 = axes[0, 0]
    for i, m in enumerate(resultados_modulos):
        ax1.plot(m['tiempo']/60, m['voltaje'], color=colores[i], 
                 label=f"M{m['modulo']}", linewidth=1)
    ax1.set_xlabel('Tiempo (minutos)')
    ax1.set_ylabel('Voltaje (V)')
    ax1.set_title('Voltaje por Módulo durante Descarga')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=4, fontsize=8)
    ax1.set_ylim([5.5, 8.5])
    
    # -------- Gráfico 2: SOC vs Tiempo --------
    ax2 = axes[0, 1]
    for i, m in enumerate(resultados_modulos):
        ax2.plot(m['tiempo']/60, m['soc']*100, color=colores[i], 
                 label=f"M{m['modulo']}", linewidth=1)
    ax2.set_xlabel('Tiempo (minutos)')
    ax2.set_ylabel('SOC (%)')
    ax2.set_title('Estado de Carga (SOC) por Módulo')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', ncol=4, fontsize=8)
    ax2.set_ylim([0, 105])
    
    # -------- Gráfico 3: SOH por Módulo (barras) --------
    ax3 = axes[1, 0]
    modulos = [m['modulo'] for m in resultados_modulos]
    sohs = [m['soh'] for m in resultados_modulos]
    colores_barras = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in sohs]
    
    barras = ax3.bar(modulos, sohs, color=colores_barras, edgecolor='black')
    ax3.axhline(y=80, color='green', linestyle='--', label='Umbral bueno (80%)')
    ax3.axhline(y=60, color='red', linestyle='--', label='Umbral crítico (60%)')
    ax3.set_xlabel('Número de Módulo')
    ax3.set_ylabel('SOH (%)')
    ax3.set_title('Estado de Salud (SOH) por Módulo')
    ax3.set_ylim([0, 110])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores sobre las barras
    for barra, soh in zip(barras, sohs):
        ax3.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
                 f'{soh:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # -------- Gráfico 4: Comparación de voltajes en t=30min --------
    ax4 = axes[1, 1]
    tiempo_analisis = 1800  # 30 minutos
    idx = np.argmin(np.abs(resultados_modulos[0]['tiempo'] - tiempo_analisis))
    voltajes_30min = [m['voltaje'][idx] for m in resultados_modulos]
    voltaje_promedio = np.mean(voltajes_30min)
    
    ax4.bar(modulos, voltajes_30min, color='steelblue', edgecolor='black')
    ax4.axhline(y=voltaje_promedio, color='red', linestyle='-', linewidth=2,
                label=f'Promedio: {voltaje_promedio:.3f} V')
    ax4.axhline(y=voltaje_promedio + 0.15, color='orange', linestyle='--', 
                label='Límite +150mV')
    ax4.axhline(y=voltaje_promedio - 0.15, color='orange', linestyle='--',
                label='Límite -150mV')
    ax4.set_xlabel('Número de Módulo')
    ax4.set_ylabel('Voltaje (V)')
    ax4.set_title(f'Comparación de Voltajes en t = 30 min')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar figura
    nombre_archivo = escenario_nombre.replace(' ', '_').replace('-', '').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
    nombre_archivo = ''.join(c for c in nombre_archivo if c.isalnum() or c == '_')
    plt.savefig(f'resultados/{nombre_archivo}.png', dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Gráfico guardado: resultados/{nombre_archivo}.png")
    
    plt.close()

# ============================================================================
# FUNCIÓN: Generar tablas de resultados
# ============================================================================

def generar_tablas(escenario_nombre, resultados_modulos, clasificacion):
    """
    Genera tablas de resultados en formato CSV y texto.
    """
    # Crear DataFrame con resultados
    datos = []
    for i, m in enumerate(resultados_modulos):
        datos.append({
            'Módulo': m['modulo'],
            'Capacidad Nominal (Ah)': CAPACIDAD_NOMINAL,
            'Capacidad Efectiva (Ah)': round(m['capacidad_efectiva'], 2),
            'Resistencia (mΩ)': round(m['resistencia'], 1),
            'SOH (%)': round(m['soh'], 1),
            'Estado': clasificacion[i]['estado'],
            'Diferencia Voltaje (mV)': round(clasificacion[i]['diferencia_mV'], 1)
        })
    
    df = pd.DataFrame(datos)
    
    # Guardar como CSV
    nombre_archivo = escenario_nombre.replace(' ', '_').replace('-', '').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
    nombre_archivo = ''.join(c for c in nombre_archivo if c.isalnum() or c == '_')
    df.to_csv(f'resultados/{nombre_archivo}_tabla.csv', index=False)
    print(f"  ✓ Tabla guardada: resultados/{nombre_archivo}_tabla.csv")
    
    # Calcular resumen del pack
    soh_promedio = np.mean([m['soh'] for m in resultados_modulos])
    soh_minimo = np.min([m['soh'] for m in resultados_modulos])
    modulos_degradados = sum(1 for c in clasificacion if c['estado'] == 'DEGRADADO')
    modulos_sospechosos = sum(1 for c in clasificacion if c['estado'] == 'SOSPECHOSO')
    
    # Diagnóstico global
    if soh_promedio >= 80 and modulos_degradados == 0:
        diagnostico = "BATERÍA EN BUEN ESTADO"
    elif soh_promedio >= 60 and modulos_degradados <= 2:
        diagnostico = "BATERÍA ACEPTABLE - Revisar módulos sospechosos"
    elif modulos_degradados <= 3 and soh_promedio >= 50:
        diagnostico = "DESBALANCE LOCALIZADO - Posible reparación"
    else:
        diagnostico = "BATERÍA MUY DEGRADADA - Recomendable reemplazo"
    
    resumen = {
        'Escenario': escenario_nombre,
        'SOH Promedio (%)': round(soh_promedio, 1),
        'SOH Mínimo (%)': round(soh_minimo, 1),
        'Módulos Sanos': NUM_MODULOS - modulos_degradados - modulos_sospechosos,
        'Módulos Sospechosos': modulos_sospechosos,
        'Módulos Degradados': modulos_degradados,
        'Diagnóstico': diagnostico
    }
    
    return df, resumen

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta la simulación completa para todos los escenarios.
    """
    print("\n" + "="*60)
    print("INICIANDO SIMULACIÓN DE DIAGNÓSTICO DE BATERÍA HÍBRIDA")
    print("="*60)
    print(f"\nConfiguración del pack:")
    print(f"  - Número de módulos: {NUM_MODULOS}")
    print(f"  - Voltaje nominal por módulo: {VOLTAJE_NOMINAL_MODULO} V")
    print(f"  - Capacidad nominal: {CAPACIDAD_NOMINAL} Ah")
    print(f"  - Corriente de descarga (C/2): {CORRIENTE_DESCARGA} A")
    print(f"  - Tiempo de simulación: {TIEMPO_TOTAL/60:.0f} minutos")
    
    # Crear escenarios
    escenarios = crear_escenarios()
    
    # Lista para almacenar todos los resúmenes
    todos_resumenes = []
    
    # Simular cada escenario
    for nombre, datos in escenarios.items():
        # Simular pack
        resultados = simular_pack(nombre, datos)
        
        # Analizar desbalance
        clasificacion, v_prom, v_std = analizar_desbalance(resultados)
        
        # Generar gráficos
        generar_graficos(nombre, resultados)
        
        # Generar tablas
        df, resumen = generar_tablas(nombre, resultados, clasificacion)
        todos_resumenes.append(resumen)
        
        # Mostrar resumen
        print(f"\n  RESUMEN DEL DIAGNÓSTICO:")
        print(f"  - SOH Promedio: {resumen['SOH Promedio (%)']}%")
        print(f"  - SOH Mínimo: {resumen['SOH Mínimo (%)']}%")
        print(f"  - Módulos degradados: {resumen['Módulos Degradados']}")
        print(f"  - DIAGNÓSTICO: {resumen['Diagnóstico']}")
    
    # Crear tabla resumen comparativa
    df_resumen = pd.DataFrame(todos_resumenes)
    df_resumen.to_csv('resultados/RESUMEN_COMPARATIVO.csv', index=False)
    print(f"\n{'='*60}")
    print("TABLA RESUMEN COMPARATIVA DE TODOS LOS ESCENARIOS")
    print("="*60)
    print(df_resumen.to_string(index=False))
    print(f"\n✓ Resumen guardado: resultados/RESUMEN_COMPARATIVO.csv")
    
    # Generar gráfico comparativo de SOH
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(todos_resumenes))
    width = 0.35
    
    soh_promedio = [r['SOH Promedio (%)'] for r in todos_resumenes]
    soh_minimo = [r['SOH Mínimo (%)'] for r in todos_resumenes]
    nombres = [r['Escenario'].replace('Escenario ', 'Esc. ').split(' - ')[0] for r in todos_resumenes]
    
    bars1 = ax.bar(x - width/2, soh_promedio, width, label='SOH Promedio', color='steelblue')
    bars2 = ax.bar(x + width/2, soh_minimo, width, label='SOH Mínimo', color='coral')
    
    ax.axhline(y=80, color='green', linestyle='--', label='Umbral Bueno (80%)')
    ax.axhline(y=60, color='red', linestyle='--', label='Umbral Crítico (60%)')
    
    ax.set_ylabel('SOH (%)')
    ax.set_title('Comparación de SOH entre Escenarios de Simulación')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.legend()
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores sobre las barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('resultados/COMPARACION_ESCENARIOS.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico comparativo guardado: resultados/COMPARACION_ESCENARIOS.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("="*60)
    print(f"\nArchivos generados en la carpeta 'resultados/':")
    print("  - 4 gráficos de escenarios (.png)")
    print("  - 4 tablas de datos (.csv)")
    print("  - 1 gráfico comparativo (.png)")
    print("  - 1 tabla resumen comparativa (.csv)")
    print("\n¡Listo para usar en tu tesis!")

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
    input("\nPresiona ENTER para cerrar...")